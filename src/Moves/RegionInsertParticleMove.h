#pragma once

#include "Move.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../ForceFields/LennardJonesTSFF.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	// Class for particle insertion move. Based on providing 
	// a list of species, the move will stash "stashcount" copies 
	// of each one, in each world in the world manager. However, 
	// if the number of stashed particles runs out (say, due to deletion)
	// then they are automatically replenished by the world, at an expense.
	class RegionInsertParticleMove : public Move
	{
	private: 
		Rand _rand;
		int _rejected;
		int _performed;
		std::vector<int> _species;
		bool _prefac;
		int _scount; // Stash count.
		bool _multi_insert;
		unsigned _seed;
		double _xlow;
		double _ylow;
		double _zlow;
		double _xhigh;
		double _yhigh;
		double _zhigh;

		void InitStashParticles(const WorldManager& wm)
		{
			// Get particle map, find one of the appropriate species 
			// and clone.
			auto& plist = Particle::GetParticleMap();
			for(auto& id : _species)
			{
				// Since the species exists, we assume there must be at least 
				// one find. 
				auto pcand = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);

				auto* P = pcand->second->Clone();

				// Stash a characteristic amount of the particles in world.
				for(auto& world : wm)
					world->StashParticle(P, _scount);
			}
		}

	public:
		RegionInsertParticleMove(const std::vector<int>& species, 
						   const WorldManager& wm,
						   int stashcount, bool multi_insert,
						   double xlow, double ylow, double zlow,
						   double xhigh, double yhigh, double zhigh,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _scount(stashcount), 
		_multi_insert(multi_insert), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
		{
			// Verify species list and add to local vector.
			auto& list = Particle::GetSpeciesList();
			for(auto& id : species)
			{
				if(id >= (int)list.size())
				{
					std::cerr << "Species ID \"" 
							  << id << "\" provided does not exist." 
							  << std::endl;
					exit(-1);
				}
				_species.push_back(id);
			}

			InitStashParticles(wm);
		}

		RegionInsertParticleMove(const std::vector<std::string>& species, 
						   const WorldManager& wm,
						   int stashcount, bool multi_insert,
						   double xlow, double ylow, double zlow,
						   double xhigh, double yhigh, double zhigh,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _scount(stashcount), 
		_multi_insert(multi_insert), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
		{
			// Verify species list and add to local vector.

			auto& list = Particle::GetSpeciesList();
			for(auto& id : species)
			{
				auto it = std::find(list.begin(), list.end(), id);
				if(it == list.end())
				{
					std::cerr << "Species ID \""
							  << id << "\" provided does not exist."
							  << std::endl;
					exit(-1);
				}
				_species.push_back(it - list.begin());
			}

			InitStashParticles(wm);
		}



		/*********************************WALL ENERGY CALCULATION **************************/
		double ConfinedWallEnergy(Particle* pi)
		{
			double epsilon = 1.0;
			double sigma = 1;
			double wall_rc = 1.00;
			double LJ_wall_E = 0.0;
			double LJ_E = 0;


			double r_y_high = fabs(pi->GetPosition()[1] - _yhigh);
			double r_y_low = fabs(pi->GetPosition()[1] - _ylow);
			double r_z_high = fabs(pi->GetPosition()[2] - _zhigh);
			double r_z_low = fabs(pi->GetPosition()[2] - _zlow);

			//std::cout <<"THE ylow is  " << r_y_low <<std::endl; 
			//std::cout <<"THE yhigh is  " << r_y_high <<std::endl; 
			//std::cout <<"THE zlow is  " << r_z_low <<std::endl; 
			//std::cout <<"THE zhigh is  " << r_z_high <<std::endl; 

			double arr[] = {r_y_high, r_y_low, r_z_high, r_z_low};
			std::vector<double> vec (arr, arr + sizeof(arr) / sizeof(arr[0]) );
			for (std::vector<int>::size_type k = 0; k != vec.size(); k++)
			{
				double r_calc = vec[k];
				if (r_calc <= wall_rc)
				{
					LJ_E = 4.0*epsilon*(pow(sigma/r_calc, 12) - pow(sigma/r_calc,6)) - 
					4.0*epsilon*(pow(sigma/wall_rc, 12) - pow(sigma/wall_rc,6));
					/*
					LJ_E = epsilon*(0.13333*pow(sigma/r_calc, 9) - pow(sigma/r_calc,3)) - 
					epsilon*(0.13333*pow(sigma/wall_rc, 9) - pow(sigma/wall_rc,3));  
					*/
					//std::cout << std::fixed <<"THE LJ_E  " << LJ_E <<std::endl;
					//std::cout << std::fixed <<"I AM BELOW OR AT WALL_RC  " << r_calc <<std::endl;
				}else
				{
					LJ_E = 0.0;
				}
				LJ_wall_E += LJ_E;
			}

			//std::cout << std::fixed <<"LJ_wall_E is  " << LJ_wall_E <<std::endl; 
			//std::cout << std::fixed <<"******************************************"<<std::endl;
			//std::cout << std::fixed <<" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
			//std::cout << std::fixed <<"                                          "<<std::endl;
			//std::cout << std::fixed <<"                                           "<<std::endl;
			return LJ_wall_E;
			// ***********WALL INTERACTION ENERGY ADDED**************************
		}



		virtual void Perform(WorldManager* wm, 
							 ForceFieldManager* ffm, 
							 const MoveOverride& override) override
		{
			auto store = 0;
			auto store_mu = 0;
			auto store_lambda = 0;
			auto store_N = 0; 

			// Get random world.
			World* w = wm->GetRandomWorld();

			Particle* plist[32];

			unsigned int NumberofParticles=1;

			// Unstash into particle list or single particle
			if(_multi_insert)
			{
				NumberofParticles = _species.size();
				for (unsigned int i = 0; i < _species.size(); i++)
					plist[i] = w->UnstashParticle(_species[i]);
			}
			else
			{
				size_t n = _species.size();
				assert(n > 0);
				auto type = _rand.int32() % n;
				plist[0] = w->UnstashParticle(_species[type]);
			}

			double Prefactor = 1;
			auto& sim = SimInfo::Instance();
			auto beta = 1.0/(sim.GetkB()*w->GetTemperature());



			// *************** NEW region based ADDITION ***********//
			// get the specified volume region
			// make sure to specify the saphron box size corrected for wall thickness in lammps, so _ylow is 1 more than where the low wall is placed
			auto V = abs(_xhigh - _xlow) * abs(_yhigh - _ylow) * abs(_zhigh - _zlow);

			// ***************************************************//




			auto& comp = w->GetComposition();
			// Get previous energy
			auto ei = ffm->EvaluateEnergy(*w);

			// Generate a random position and orientation for particle insertion.
			for (unsigned int i = 0; i < NumberofParticles; i++)
			{
				const auto& H = w->GetHMatrix();
				Vector3D pr{_rand.doub(), _rand.doub(), _rand.doub()};





				// *************** NEW region based  ***********//
				// this new region can be cubic or rectangular
				// this can be used for confinement too but have to specify 6 dimensions
				// even though evaluate energy and evaluate pressure takes into account world volume, since delta E and P, so it doesn't seem to matter
				// and delta E and P should accurately take into account delta E due to particle insertion 

				// define a new matrix
				Matrix3D H_store(arma::fill::zeros);
				H_store(0,0) = abs(_xhigh - _xlow);
				H_store(1,1) = abs(_yhigh - _ylow);
				H_store(2,2) = abs(_zhigh - _zlow);

				// create a new vector shift that will be added elementwise
				Vector3D shift{_xlow, _ylow, _zlow};

				// the pos of the particle is random position inside confinement. shift is there because coordinate
				// is defined wrt to the Box size not the confinement size.
				Vector3D pos = H_store*pr + shift;

				// ***************************************************//





				plist[i]->SetPosition(pos);

				// Choose random axis, and generate random angle.
				int axis = _rand.int32() % 3 + 1;
				double deg = (4.0*_rand.doub() - 2.0)*M_PI;
				Matrix3D R = GenRotationMatrix(axis, deg);

				// Rotate particle and director.
				plist[i]->SetDirector(R*plist[i]->GetDirector());
				for(auto& child : *plist[i])
				{
					child->SetPosition(R*(child->GetPosition()-pos) + pos);
					child->SetDirector(R*child->GetDirector());
				}

				auto id = plist[i]->GetSpeciesID();
				auto N = comp[id];






				// *************** NEW CONFINEMENT ADDITION ***********//

				auto mu = w->GetChemicalPotential(id);
				// you can use above statement but in that case the chemical pot set in the json script
				// is the chemical potential of species in the region
				
				//auto mu = w->SetChemicalPotential(id, mu_set); // this is assuming the pair on which this move is performed is same
																// this way we can set region based chemical potential

				// ***************************************************//






				auto lambda = w->GetWavelength(id);
				store_lambda = lambda;
				store_mu = mu;
				store_N = N; 
				w->AddParticle(plist[i]);
				Prefactor*=V/(lambda*lambda*lambda*(N+1))*exp(beta*mu);
				store = i;

			}
			auto ef = ffm->EvaluateEnergy(*w);
			++_performed;





			/* CALCULATION OF WALL ENERGY*/
			// ***********WALL INTERACTION ENERGY ADDED**************************
			//IMP:- THIS SECTION IS ONLY FOR POLYMER CONFINEMENT SIMULATION WHERE YOU ARE CONFINING BY lj/wall, BECAUSE OF THAT 
			//YOU HAVE TO ADD ADDITIONAL ENERGY TO PACC EQUATION. IF THERE IS NO lj/wall THEN THIS SECTION OF WALL INTERACTION ENERGY 
			//IS NOT NEEDED !!
			//HENCE THIS WILL BE COMMENTED OUT MOST OF THE TIME UNLESS YOU ARE DOING THIS SPECIFIC CASE FOR GCMC SIMULATION
			//NOTE ALSO THAT YOU WILL HAVE TO TAKE CARE WHAT KIND OF WALL DID YOU PUT IN, I.E. IS IT 9-3, 12-6 AND ACCORDINGLY
			//CHANGE THE PARAMETERS EPSILON RC ETC !!!
			double WALL_E = 0;
			for (unsigned int i = 0; i < NumberofParticles; i++)
			{
				WALL_E += ConfinedWallEnergy(plist[i]);
			}
			// ***********WALL INTERACTION ENERGY ADDED**************************






			// The acceptance rule is from Frenkel & Smit Eq. 5.6.8.
			// However,t iwas modified since we are using the *final* particle number.
			auto de = ef - ei;
			//auto pacc = Prefactor*exp(-beta*de.energy.total());
			auto pacc = Prefactor*exp(-beta*(de.energy.total()+WALL_E)); // PACC WITH WALL ENERGY ADDED IN
			pacc = pacc > 1.0 ? 1.0 : pacc;

			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				// Stashing a particle automatically removes it from world.
				for (unsigned int i = 0; i < NumberofParticles; i++)
					w->StashParticle(plist[i]);

				++_rejected;
			}
			else
			{
				// Update energies and pressures.
				w->SetEnergy(ef.energy);
				w->SetPressure(ef.pressure);
				/*
				std::cout << " THIS IS INSERT MOVE "<<std::endl; 
				std::cout << " PACC IS: " << pacc <<std::endl;		
				std::cout << " total energy is:  " << ef.energy.total() <<std::endl; 
				std::cout << " id: " << plist[store]->GetSpeciesID() <<std::endl; 
				std::cout << " Prefactor: " << Prefactor <<std::endl; 
				std::cout << " Volume: " << V <<std::endl; 
				std::cout << " lambda: " << store_lambda <<std::endl; 
				std::cout << " mu: " << store_mu <<std::endl; 
				std::cout << " N: " << store_N <<std::endl; 
				std::cout <<" xlo " << _xlow <<std::endl; 
				std::cout <<" xhi " << _xhigh <<std::endl; 
				std::cout <<" ylo " << _ylow <<std::endl; 
				std::cout <<" yhi " << _yhigh <<std::endl; 
				std::cout <<" zlo " << _zlow <<std::endl; 
				std::cout <<" zhi " << _zhigh <<std::endl; 
				std::cout <<" particle_position " << plist[store]->GetPosition() <<std::endl; 
				std::cout <<" NumberofParticles " << NumberofParticles <<std::endl; 
				std::cout << " ******************** "<<std::endl; 
				std::cout << "                       "<<std::endl; 
				*/
			}
		}

		virtual void Perform(World* w, 
							 ForceFieldManager* ffm, 
							 DOSOrderParameter* op, 
							 const MoveOverride& override) override
		{

			Particle* plist[32];
			//Evaluate initial energy and order parameter
			auto ei = w->GetEnergy();
			auto opi = op->EvaluateOrderParameter(*w);

			unsigned int NumberofParticles=1;

			// Unstash into particle list or single particle
			if(_multi_insert)
			{
				NumberofParticles=_species.size();
				for (unsigned int i = 0; i < _species.size(); i++)
					plist[i] = w->UnstashParticle(_species[i]);
			}
			else
			{
				size_t n = _species.size();
				assert(n > 0);
				auto type = _rand.int32() % n;
				plist[0] = w->UnstashParticle(_species[type]);
			}

			double Prefactor = 1;
			auto& sim = SimInfo::Instance();
			auto beta = 1.0/(sim.GetkB()*w->GetTemperature());
			auto V = w->GetVolume();
			EPTuple ef;
			auto& comp = w->GetComposition();

			// Generate a random position and orientation for particle insertion.
			for (unsigned int i = 0; i < NumberofParticles; i++)
			{

				const auto& H = w->GetHMatrix();
				Vector3D pr{_rand.doub(), _rand.doub(), _rand.doub()};
				Vector3D pos = H*pr;
				plist[i]->SetPosition(pos);

				// Choose random axis, and generate random angle.
				int axis = _rand.int32() % 3 + 1;
				double deg = (4.0*_rand.doub() - 2.0)*M_PI;
				Matrix3D R = GenRotationMatrix(axis, deg);

				// Rotate particle and director.
				plist[i]->SetDirector(R*plist[i]->GetDirector());
				for(auto& child : *plist[i])
				{
					child->SetPosition(R*(child->GetPosition()-pos) + pos);
					child->SetDirector(R*child->GetDirector());
				}

				auto id = plist[i]->GetSpeciesID();
				auto N = comp[id];
				auto mu = w->GetChemicalPotential(id);
				auto lambda = w->GetWavelength(id);

				if(_prefac)
					Prefactor*=V/(lambda*lambda*lambda*N)*exp(beta*mu);
			
				// Evaluate new energy for each particle. 
				// Insert particle one at a time. Done this way
				// to prevent double counting in the forcefieldmanager
				// Can be adjusted later if wated.

				w->AddParticle(plist[i]);
				//ef += ffm->EvaluateEnergy(*plist[i]);
				ef += ffm->EvaluateEnergy(*w);
			}

			++_performed;

			// new energy update and eval OP. 
			w->IncrementEnergy(ef.energy);
			w->IncrementPressure(ef.pressure);
			auto opf = op->EvaluateOrderParameter(*w);

			// The acceptance rule is from Frenkel & Smit Eq. 5.6.8.
			// However, it iwas modified since we are using the *final* particle number.
			double pacc = op->AcceptanceProbability(ei, ef.energy, opi, opf, *w);
			
			// If prefactor is enabled, compute.
			if(_prefac)
				pacc *= Prefactor;

			pacc = pacc > 1.0 ? 1.0 : pacc;

			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				// Stashing a particle automatically removes it from world. 
				for (unsigned int i = 0; i < NumberofParticles; i++)
					w->StashParticle(plist[i]);

				w->IncrementEnergy(-1.0*ef.energy);
				w->IncrementPressure(-1.0*ef.pressure);
				++_rejected;
			}
			
		}

		// Turn on or off the acceptance rule prefactor 
		// for DOS order parameter.
		void SetOrderParameterPrefactor(bool flag) { _prefac = flag; }

		virtual double GetAcceptanceRatio() const override
		{
			return 1.0-(double)_rejected/_performed;
		};

		virtual void ResetAcceptanceRatio() override
		{
			_performed = 0;
			_rejected = 0;
		}

		// Serialize.
		virtual void Serialize(Json::Value& json) const override
		{
			json["type"] = "InsertParticle";
			json["stash_count"] = _scount;
			json["multi_insertion"] = _multi_insert;
			json["seed"] = _seed;
			json["op_prefactor"] = _prefac;

			auto& species = Particle::GetSpeciesList();
			for(auto& s : _species)
				json["species"].append(species[s]);
		}

		virtual std::string GetName() const override { return "InsertParticle"; }

		// Clone move.
		Move* Clone() const override
		{
			return new RegionInsertParticleMove(static_cast<const RegionInsertParticleMove&>(*this));
		}

		~RegionInsertParticleMove(){}
	};
}