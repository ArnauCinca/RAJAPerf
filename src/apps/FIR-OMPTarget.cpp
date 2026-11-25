//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define FIR_DATA_SETUP_OMP_TARGET \
  Real_ptr coeff; \
  \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocData(DataSpace::OmpTarget, coeff, FIR_COEFFLEN); \
  copyData(DataSpace::OmpTarget, coeff, DataSpace::Host, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_OMP_TARGET \
  deallocData(DataSpace::OmpTarget, coeff);


void FIR::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIR_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      //#pragma omp target teams distribute parallel for thread_limit(threads_per_team) is_device_ptr(in, out, coeff) device( did )
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend - ibegin, threads_per_team);
      #pragma omp target teams num_teams(dims(3): grid_size, 1, 1) \
              thread_limit(dims(3): threads_per_team, 1, 1) \
              is_device_ptr(in, out, coeff) device( did )
      #pragma omp parallel
      {
        Index_type i = ibegin + omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        if (i < iend) {
          FIR_BODY;
        }
      }
      RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == Base_OpenMPTarget_ParallelFor ) {

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend - ibegin, threads_per_team);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      #pragma omp target is_device_ptr(in, out, coeff) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
         FIR_BODY;
      }
      RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    FIR_COEFF;

    FIR_DATA_SETUP_OMP_TARGET;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        FIR_BODY;
      });
      RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_OMP_TARGET;

  } else {
     getCout() << "\n  FIR : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FIR, OpenMPTarget, Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor, RAJA_OpenMPTarget)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
