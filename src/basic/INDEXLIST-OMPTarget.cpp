//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP) \
 && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#endif


void INDEXLIST::runOpenMPTargetVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP) \
 && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

    // Both variants require a worksharing loop for OpenMP scan.
    case Base_OpenMPTarget_ParallelFor :
    case Base_OpenMPTarget : {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend - ibegin, threads_per_team);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_1");
        Index_type count = 0;
        //#pragma omp target is_device_ptr(x, list) device( did )
        //#pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) \
        //                                          reduction(inscan, +:count)
        // OpenMP scan requires an associated worksharing loop.
        #pragma omp target teams distribute parallel for num_teams(grid_size) thread_limit(threads_per_team) \
                                                  reduction(inscan, +:count) \
                                                  is_device_ptr(x, list) device( did )
        for (Index_type i = ibegin; i < iend; ++i ) {
          Index_type inc = 0;
          if (INDEXLIST_CONDITIONAL) {
            list[count] = i ;
            inc = 1;
          }
          #pragma omp scan exclusive(count)
          count += inc;
        }

        m_len = count;
        RP_CALI_SUBKERNEL_END("INDEXLIST_1");

      }
      stopTimer();

      break;
    }

    default : {
      ignore_unused(run_reps, ibegin, iend, x, list);
      getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP) \
 && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INDEXLIST, OpenMPTarget, Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor)
#else
void INDEXLIST::defineOpenMPTargetVariantTunings() {}
#endif

} // end namespace basic
} // end namespace rajaperf
