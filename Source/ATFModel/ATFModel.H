#ifndef ATFMODEL_H
#define ATFMODEL_H

#include <AMReX_ParmParse.H>
#include <AMReX_Gpu.H>
#include <sundials/sundials_context.h>
#include "Factory.H"
#include "PelePhysics.H"
#include "mechanism.H"

class ATFModel
{

public: 
//
// Default constructur
ATFModel(amrex::Real max_thickening_factor, bool do_sensor, bool do_efficiency);


// compute thickening factors

amrex::Real max_thickening;
bool do_flame_sensor;
bool do_efficiency_function;

void computeFlameSensors(
    const amrex::Box& bx, 
    const amrex::Array4<const amrex::Real> progress_variables,
    amrex::Array4<amrex::Real> flame_sensors);

void computeThickeningFactors(
    const amrex::Box& bx, 
    const amrex::Array4<const amrex::Real> flame_sensors,
    amrex::Array4<amrex::Real> thickening_factors);

void computeEfficiencyFunctions(
    const amrex::Box& bx, 
    amrex::Array4<amrex::Real> thickening_factors,
    amrex::Array4<amrex::Real> efficiency_functions);


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
static amrex::Real calculateOmega(const amrex::Real pv) noexcept
{
    if (pv < 0.0 || pv > 1.0) return 0.0;
    const amrex::Real t = pv * (1.0 - pv);
    return 16.0 * t * t;
}

};

#endif
