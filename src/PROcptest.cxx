#include "PROcptest.h"
#include "PROcess.h"
#include "PROmodel.h"
#include "PROlog.h"

#include <cmath>
#include <limits>
#include <random>

namespace PROfit {

// Build a PROmetric for the given data, config, and chi2 type.
// Caller takes ownership (must delete).
static PROmetric* make_metric(const std::string& chi2,
                               const PROconfig& config,
                               const PROpeller& prop,
                               const PROsyst& systs,
                               PROmodel& model,
                               const PROdata& data,
                               PROmetric::EvalStrategy strat,
                               bool shapeonly)
{
    if(chi2 == "PROchi")
        return new PROchi("", config, prop, &systs, model, data, strat, shapeonly);
    if(chi2 == "PROCNP")
        return new PROCNP("", config, prop, &systs, model, data, strat, shapeonly);
    if(chi2 == "Poisson")
        return new PROpoisson("", config, prop, &systs, model, data, strat, shapeonly);
    log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
    abort();
}

void cptest_worker(cptest_args args) {
    log<LOG_INFO>(L"%1% || CP test worker thread %2% : %3% phi_true points")
        % __func__ % args.thread_id % args.phi_vals.size();

    std::mt19937 rng{args.seed};
    std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());

    std::unique_ptr<PROmodel> model = get_model_from_string(args.config, args.prop);

    PROmetric::EvalStrategy strat = args.eventbyevent
        ? PROmetric::EventByEvent
        : PROmetric::BinnedChi2;

    // Lambda: generate Asimov data + run one constrained fit.
    // Returns {chi2, best_fit_params}.
    // warm_seed: previous phi_true's best-fit (empty VectorXf if first point).
    auto one_fit = [&](float phi_true, float phi_fixed,
                       const Eigen::VectorXf& warm_seed)
                   -> std::pair<float, Eigen::VectorXf>
    {
        // --- Asimov data for this phi_true ---
        Eigen::VectorXf inject = args.inject_params;
        inject(args.phi54_idx) = phi_true;
        PROspec data_spec = FillSpectra(args.config, args.prop, args.systs, *model,
                                        inject, true, args.config.i_prime);
        Eigen::VectorXf data_vec = CollapseMatrix(args.config, data_spec.Spec(),
                                                   (int)args.config.i_prime);
        PROdata data(data_vec, data_vec.array().sqrt());

        // --- Constrained bounds (phi54 fixed) ---
        Eigen::VectorXf lb_local = args.lb;
        Eigen::VectorXf ub_local = args.ub;
        lb_local(args.phi54_idx) = phi_fixed;
        ub_local(args.phi54_idx) = phi_fixed;

        PROmetric* metric = make_metric(args.chi2, args.config, args.prop,
                                         args.systs, *model, data, strat, args.shapeonly);
        metric->setBounds(lb_local, ub_local);

        // --- Seeds: CV start + warm-start from previous phi_true ---
        Eigen::VectorXf cv_start = args.cv_params;
        cv_start(args.phi54_idx) = phi_fixed;
        std::vector<Eigen::VectorXf> seeds = {cv_start};
        if(warm_seed.size() > 0) {
            Eigen::VectorXf w = warm_seed;
            w(args.phi54_idx) = phi_fixed;  // enforce constraint on the warm seed
            seeds.push_back(w);
        }

        PROfitter fitter(ub_local, lb_local, args.fitconfig, dseed(rng));
        float chi2 = fitter.Fit(*metric, seeds);
        Eigen::VectorXf best = fitter.BestFit();
        delete metric;

        return {chi2, best};
    };

    const size_t N = args.phi_vals.size();
    std::vector<cptest_result> results(N);

    // ----------------------------------------------------------------
    // Forward pass: phi[0] -> phi[N-1] with warm-start carrying forward
    // ----------------------------------------------------------------
    Eigen::VectorXf warm_phi0, warm_phipi;

    for(size_t i = 0; i < N; ++i) {
        float phi_true = args.phi_vals[i];
        results[i].phi_true = phi_true;

        log<LOG_INFO>(L"%1% || Thread %2% [fwd] phi_true = %3%")
            % __func__ % args.thread_id % phi_true;

        auto [c0, b0]   = one_fit(phi_true, 0.0f,      warm_phi0);
        auto [cpi, bpi] = one_fit(phi_true, (float)M_PI, warm_phipi);

        results[i].chi2_phi0   = c0;
        results[i].chi2_phipi  = cpi;
        warm_phi0  = b0;
        warm_phipi = bpi;

        log<LOG_INFO>(L"%1% || Thread %2% [fwd] phi_true=%3% : chi2(0)=%4% chi2(pi)=%5%")
            % __func__ % args.thread_id % phi_true % c0 % cpi;
    }

    // ----------------------------------------------------------------
    // Backward pass: phi[N-1] -> phi[0], warm-start from last fwd best.
    // Update result if a lower chi2 is found.
    // ----------------------------------------------------------------
    // warm_phi0 / warm_phipi now hold the forward-pass best-fits at phi[N-1].

    for(int i = (int)N - 1; i >= 0; --i) {
        float phi_true = args.phi_vals[i];

        log<LOG_INFO>(L"%1% || Thread %2% [bwd] phi_true = %3%")
            % __func__ % args.thread_id % phi_true;

        auto [c0, b0]   = one_fit(phi_true, 0.0f,        warm_phi0);
        auto [cpi, bpi] = one_fit(phi_true, (float)M_PI,  warm_phipi);

        if(c0 < results[i].chi2_phi0) {
            log<LOG_INFO>(L"%1% || Thread %2% [bwd] improved phi=0   at phi_true=%3%: %4% -> %5%")
                % __func__ % args.thread_id % phi_true % results[i].chi2_phi0 % c0;
            results[i].chi2_phi0  = c0;
        }
        if(cpi < results[i].chi2_phipi) {
            log<LOG_INFO>(L"%1% || Thread %2% [bwd] improved phi=pi  at phi_true=%3%: %4% -> %5%")
                % __func__ % args.thread_id % phi_true % results[i].chi2_phipi % cpi;
            results[i].chi2_phipi = cpi;
        }

        // Always carry the better best-fit as warm-start for the next backward step
        warm_phi0  = b0;
        warm_phipi = bpi;
    }

    // ----------------------------------------------------------------
    // Push sorted results
    // ----------------------------------------------------------------
    for(auto& r : results) {
        log<LOG_INFO>(L"%1% || Thread %2% FINAL phi_true=%3% : chi2(0)=%4% chi2(pi)=%5%")
            % __func__ % args.thread_id % r.phi_true % r.chi2_phi0 % r.chi2_phipi;
        args.out->push_back(r);
    }
}

} // namespace PROfit
