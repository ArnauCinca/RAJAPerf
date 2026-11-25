//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

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

void POLYBENCH_3MM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_1");
      //#pragma omp target is_device_ptr(A,B,E) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      const size_t grid_size_x = RAJA_DIVIDE_CEILING_INT(nj, threads_per_team_x);
      const size_t grid_size_y = RAJA_DIVIDE_CEILING_INT(ni, threads_per_team_y);
      #pragma omp target teams num_teams(dims(3): grid_size_x, grid_size_y, 1) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, 1) \
              is_device_ptr(A,B,E) device( did )
      #pragma omp parallel
      {
        Index_type j = omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type i = omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        if (j < nj && i < ni) {
          POLYBENCH_3MM_BODY1;
          for(Index_type k = 0; k < nk; k++) {
            POLYBENCH_3MM_BODY2;
          }
          POLYBENCH_3MM_BODY3;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_2");
      //#pragma omp target is_device_ptr(C,D,F) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      const size_t grid_size2_x = RAJA_DIVIDE_CEILING_INT(nl, threads_per_team_x);
      const size_t grid_size2_y = RAJA_DIVIDE_CEILING_INT(nj, threads_per_team_y);
      #pragma omp target teams num_teams(dims(3): grid_size2_x, grid_size2_y, 1) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, 1) \
              is_device_ptr(C,D,F) device( did )
      #pragma omp parallel
      {
        Index_type l = omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type j = omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        if (l < nl && j < nj) {
          POLYBENCH_3MM_BODY4;
          for(Index_type m = 0; m < nm; m++) {
            POLYBENCH_3MM_BODY5;
          }
          POLYBENCH_3MM_BODY6;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_3");
      //#pragma omp target is_device_ptr(E,F,G) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      const size_t grid_size3_x = RAJA_DIVIDE_CEILING_INT(nl, threads_per_team_x);
      const size_t grid_size3_y = RAJA_DIVIDE_CEILING_INT(ni, threads_per_team_y);
      #pragma omp target teams num_teams(dims(3): grid_size3_x, grid_size3_y, 1) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, 1) \
              is_device_ptr(E,F,G) device( did )
      #pragma omp parallel
      {
        Index_type l = omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        Index_type i = omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        if (l < nl && i < ni) {
          POLYBENCH_3MM_BODY7;
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY8;
          }
          POLYBENCH_3MM_BODY9;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_3");

    }
    stopTimer();

  } else if ( vid == Base_OpenMPTarget_ParallelFor ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(ni) * nj, threads_per_team);
    const size_t grid_size2 = RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(nj) * nl, threads_per_team);
    const size_t grid_size3 = RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(ni) * nl, threads_per_team);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_1");
      #pragma omp target is_device_ptr(A,B,E) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < ni; i++ ) {
        for(Index_type j = 0; j < nj; j++) {
          POLYBENCH_3MM_BODY1;
          for(Index_type k = 0; k < nk; k++) {
            POLYBENCH_3MM_BODY2;
          }
          POLYBENCH_3MM_BODY3;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_2");
      #pragma omp target is_device_ptr(C,D,F) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size2) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(2)
      for(Index_type j = 0; j < nj; j++) {
        for(Index_type l = 0; l < nl; l++) {
          POLYBENCH_3MM_BODY4;
          for(Index_type m = 0; m < nm; m++) {
            POLYBENCH_3MM_BODY5;
          }
          POLYBENCH_3MM_BODY6;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_3");
      #pragma omp target is_device_ptr(E,F,G) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size3) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(2)
      for(Index_type i = 0; i < ni; i++) {
        for(Index_type l = 0; l < nl; l++) {
          POLYBENCH_3MM_BODY7;
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY8;
          }
          POLYBENCH_3MM_BODY9;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_3");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_1");
      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type k,
             Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j,
             Real_type &dot) {
          POLYBENCH_3MM_BODY3_RAJA;
        }

      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_2");
      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nm}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] (Index_type j, Index_type l, Index_type m,
             Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=] (Index_type j, Index_type l,
             Real_type &dot) {
          POLYBENCH_3MM_BODY6_RAJA;
        }

      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_3MM_3");
      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type j,
             Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] (Index_type i, Index_type l,
             Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_3MM_3");

    }
    stopTimer();

  } else {
     getCout() << "\n  POLYBENCH_3MM : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_3MM, OpenMPTarget, Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
