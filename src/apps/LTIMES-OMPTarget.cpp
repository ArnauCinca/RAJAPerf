//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

using namespace ltimes_idx;

  const size_t threads_per_team = 256;
  // Match the factorized CUDA layout: moments (x), groups (y), zones (z).
  const size_t threads_per_team_x = 32;
  const size_t threads_per_team_y = integer::greater_of_squarest_factor_pair(threads_per_team / threads_per_team_x);
  const size_t threads_per_team_z = integer::lesser_of_squarest_factor_pair(threads_per_team / threads_per_team_x);

void LTIMES::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      //#pragma omp target firstprivate(psi, ell, phi, num_z, num_g, num_m, num_d) device( did )
      //#pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      const size_t grid_size_x = RAJA_DIVIDE_CEILING_INT(*num_m, threads_per_team_x);
      const size_t grid_size_y = RAJA_DIVIDE_CEILING_INT(*num_g, threads_per_team_y);
      const size_t grid_size_z = RAJA_DIVIDE_CEILING_INT(*num_z, threads_per_team_z);
      #pragma omp target teams num_teams(dims(3): grid_size_x, grid_size_y, grid_size_z) \
              thread_limit(dims(3): threads_per_team_x, threads_per_team_y, threads_per_team_z) \
              firstprivate(psi, ell, phi, num_z, num_g, num_m, num_d) device( did )
      #pragma omp parallel
      {
        RAJA::Index_type im = omp_get_team_num_dim(0) * omp_get_num_threads_dim(0) + omp_get_thread_num_dim(0);
        RAJA::Index_type ig = omp_get_team_num_dim(1) * omp_get_num_threads_dim(1) + omp_get_thread_num_dim(1);
        RAJA::Index_type iz = omp_get_team_num_dim(2) * omp_get_num_threads_dim(2) + omp_get_thread_num_dim(2);
        if (im < *num_m && ig < *num_g && iz < *num_z) {
          IZ z(iz);
          IG g(ig);
          IM m(im);
          for (ID d(0); d < num_d; ++d ) {
            LTIMES_BODY;
          }
        }
      }
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else if ( vid == Base_OpenMPTarget_ParallelFor ) {

    const size_t grid_size =
        RAJA_DIVIDE_CEILING_INT(static_cast<size_t>(*num_z) * (*num_g) * (*num_m), threads_per_team);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      #pragma omp target firstprivate(psi, ell, phi, num_z, num_g, num_m, num_d) device( did )
      #pragma omp teams distribute parallel for num_teams(grid_size) \
              thread_limit(threads_per_team) schedule(static, 1) collapse(3)
      for (RAJA::Index_type iz = 0; iz < *num_z; ++iz ) {
        for (RAJA::Index_type ig = 0; ig < *num_g; ++ig ) {
          for (RAJA::Index_type im = 0; im < *num_m; ++im ) {
            IZ z(iz);
            IG g(ig);
            IM m(im);
            for (ID d(0); d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<1, 2, 3>, // z, g, m
          RAJA::statement::For<0, RAJA::seq_exec,         // d
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(IDRange(0, *num_d),
                                                        IZRange(0, *num_z),
                                                        IGRange(0, *num_g),
                                                        IMRange(0, *num_m)),
        res,
        [=] (ID d, IZ z, IG g, IM m) {
        LTIMES_BODY;
      });
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else {
     getCout() << "\n LTIMES : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

void LTIMES::defineOpenMPTargetVariantTunings()
{

  for (VariantID vid : {Base_OpenMPTarget, Base_OpenMPTarget_ParallelFor, RAJA_OpenMPTarget}) {

    if (vid == RAJA_OpenMPTarget) {

      addVariantTuning<&LTIMES::runOpenMPTargetVariant>(
        vid, "kernel");

    } else {

      addVariantTuning<&LTIMES::runOpenMPTargetVariant>(
        vid, "default");

    }

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
