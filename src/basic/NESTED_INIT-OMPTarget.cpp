//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


  const size_t threads_per_team = 256;
  // Match the CUDA block shape, with the innermost index along x.
  const size_t threads_per_team_x = 32;
  const size_t threads_per_team_y = threads_per_team / threads_per_team_x;
  const size_t threads_per_team_z = 1;

void NESTED_INIT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      //#pragma omp target is_device_ptr(array) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      const size_t grid_size_x = RAJA_DIVIDE_CEILING_INT(ni, threads_per_team_x);
      const size_t grid_size_y = RAJA_DIVIDE_CEILING_INT(nj, threads_per_team_y);
      const size_t grid_size_z = RAJA_DIVIDE_CEILING_INT(nk, threads_per_team_z);
      #pragma omp target teams num_teams(dims(3): grid_size_x, grid_size_y, grid_size_z) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, threads_per_team_z) \
              is_device_ptr(array) device( did )
      #pragma omp parallel
      {
        Index_type i = omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type j = omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        Index_type k = omp_get_team_num_dim(2);
        if (i < ni && j < nj && k < nk) {
          NESTED_INIT_BODY;
        }
      }
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else if ( vid == Base_OpenMPTarget_ParallelFor ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(nk) * nj * ni, threads_per_team);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      #pragma omp target is_device_ptr(array) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(3)
      for (Index_type k = 0; k < nk; ++k ) {
        for (Index_type j = 0; j < nj; ++j ) {
          for (Index_type i = 0; i < ni; ++i ) {
            NESTED_INIT_BODY;
          }
        }
      }
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<2, 1, 0>, // k, j, i
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::kernel_resource<EXEC_POL>(
         RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                          RAJA::RangeSegment(0, nj),
                          RAJA::RangeSegment(0, nk)),
         res,
         [=](Index_type i, Index_type j, Index_type k) {
           NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(NESTED_INIT, OpenMPTarget, Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor, RAJA_OpenMPTarget)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
