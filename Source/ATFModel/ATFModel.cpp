
// AMReX include statements
#include <AMReX_Reduce.H>

// PelePhysics include statements
#include "PelePhysics.H"

// ATFModel include statements
#include "ATFModel.H"

// Default constructor

ATFModel::ATFModel(
amrex::Real max_thickening_factor,
bool do_sensor,
bool do_efficiency){
max_thickening = max_thickening_factor;
do_flame_sensor = do_sensor;
do_efficiency_function = do_efficiency;
}

void ATFModel::computeFlameSensors(
    const amrex::Box& bx, 
    const amrex::Array4<const amrex::Real> progress_variables,
    amrex::Array4<amrex::Real> flame_sensors){

        amrex::ParallelFor(bx, [progress_variables, flame_sensors] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {  
            amrex::Real PV = progress_variables(i, j, k);
            amrex::Real omega = ATFModel::calculateOmega(PV);
            flame_sensors(i, j, k) = omega;
        });
    }

void ATFModel::computeThickeningFactors(
    const amrex::Box& bx, 
    const amrex::Array4<const amrex::Real> flame_sensors,
    amrex::Array4<amrex::Real> thickening_factors){

	bool use_sensor = do_flame_sensor;
	amrex::Real max_thick = max_thickening;
	
	AMREX_ASSERT(flame_sensors.contains(bx));
	AMREX_ASSERT(thickening_factors.contains(bx));
        
        amrex::ParallelFor(bx, [flame_sensors, thickening_factors, use_sensor, max_thick] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {  
            if(use_sensor){      
                amrex::Real omega = flame_sensors(i, j, k);
                amrex::Real thickening_factor = 1 + (max_thick - 1) * omega;
                thickening_factors(i,j,k) = thickening_factor;
            }else{
                thickening_factors(i,j,k) = max_thick;
            }
        });
      
}

void ATFModel::computeEfficiencyFunctions(
    const amrex::Box& bx, 
    amrex::Array4<amrex::Real> thickening_factors,
    amrex::Array4<amrex::Real> efficiency_functions){

	bool use_efficiency_function = do_flame_sensor;

        amrex::ParallelFor(bx, [thickening_factors, efficiency_functions, use_efficiency_function] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {        
            if(use_efficiency_function){
                amrex::Real efficiency_function = std::pow(thickening_factors(i,j,k), 2.0/3.0);
                efficiency_functions(i,j,k) = efficiency_function;
            }else{
                efficiency_functions(i,j,k) = 1;
            }
        });
}

