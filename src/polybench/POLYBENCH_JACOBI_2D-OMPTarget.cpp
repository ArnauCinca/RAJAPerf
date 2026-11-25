//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  const size_t threads_per_team = 256;
  // Match the CUDA block shape, with the innermost index along x.
  const size_t threads_per_team_x = 32;
  const size_t threads_per_team_y = threads_per_team / threads_per_team_x;

void POLYBENCH_JACOBI_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      //#pragma omp target is_device_ptr(A,B) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      const size_t grid_size_x = RAJA_DIVIDE_CEILING_INT(N-2, threads_per_team_x);
      const size_t grid_size_y = RAJA_DIVIDE_CEILING_INT(N-2, threads_per_team_y);
      #pragma omp target teams num_teams(dims(3): grid_size_x, grid_size_y, 1) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, 1) \
              is_device_ptr(A,B) device( did )
      #pragma omp parallel
      {
        Index_type j = 1 + omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type i = 1 + omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        if (j < N-1 && i < N-1) {
          POLYBENCH_JACOBI_2D_BODY1;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      //#pragma omp target is_device_ptr(A,B) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      const size_t grid_size2_x = RAJA_DIVIDE_CEILING_INT(N-2, threads_per_team_x);
      const size_t grid_size2_y = RAJA_DIVIDE_CEILING_INT(N-2, threads_per_team_y);
      #pragma omp target teams num_teams(dims(3): grid_size2_x, grid_size2_y, 1) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, 1) \
              is_device_ptr(A,B) device( did )
      #pragma omp parallel
      {
        Index_type j = 1 + omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type i = 1 + omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        if (j < N-1 && i < N-1) {
          POLYBENCH_JACOBI_2D_BODY2;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else if ( vid == Base_OpenMPTarget_ParallelFor ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(N - 2) * (N - 2), threads_per_team);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(2)
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          POLYBENCH_JACOBI_2D_BODY1;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(2)
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          POLYBENCH_JACOBI_2D_BODY2;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget) {

    auto res{getOmpTargetResource()};

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_JACOBI_2D_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_JACOBI_2D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_2D, OpenMPTarget, Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
