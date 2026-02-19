#ifndef PRO_CPTEST_H
#define PRO_CPTEST_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROmetric.h"
#include "PROtocall.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"

#include <Eigen/Eigen>
#include <vector>
#include <string>

namespace PROfit {

    // Result for one phi_true point
    struct cptest_result {
        float phi_true;
        float chi2_phi0;    // chi2 with phi54 fixed at 0
        float chi2_phipi;   // chi2 with phi54 fixed at pi
    };

    // Arguments passed to each worker thread
    struct cptest_args {
        size_t phi54_idx;                   // Index of phi54 in parameter vector
        std::vector<float> phi_vals;        // Subset of phi_true values for this thread
        std::vector<cptest_result>* out;    // Thread-local results (no mutex needed)
        const PROconfig config;
        const PROpeller prop;
        const PROsyst systs;
        std::string chi2;
        const Eigen::VectorXf inject_params; // True oscillation params; phi54 overridden per point
        const Eigen::VectorXf cv_params;     // Starting point for fit (CVParams)
        const Eigen::VectorXf lb;            // Lower bounds (global_lb); phi54 overridden per fit
        const Eigen::VectorXf ub;            // Upper bounds (global_ub); phi54 overridden per fit
        PROfitterConfig fitconfig;
        uint32_t seed;
        int thread_id;
        bool eventbyevent;
        bool shapeonly;
    };

    void cptest_worker(cptest_args args);

} // namespace PROfit

#endif
