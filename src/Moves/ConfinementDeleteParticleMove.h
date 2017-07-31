#pragma once

#include "Move.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	// Class for particle deletion move.
	class ConfinementDeleteParticleMove : public Move
	{
	private: 
		Rand _rand;
		int _rejected;
		int _performed;
		std::vector<int> _species;
		bool _prefac;
		bool _multi_delete;
		unsigned _seed;
		double _xlow;
		double _ylow;
		double _zlow;
		double _xhigh;
		double _yhigh;
		double _zhigh;

	public:
		ConfinementDeleteParticleMove(const std::vector<int>& species, 
						   bool multi_delete,
   						   double xlow, double ylow, double zlow,
						   double xhigh, double yhigh, double zhigh,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _multi_delete(multi_delete), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
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
		}

		ConfinementDeleteParticleMove(const std::vector<std::string>& species,
						   bool multi_delete,
   						   double xlow, double ylow, double zlow,
						   double xhigh, double yhigh, double zhigh,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _multi_delete(multi_delete), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
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
		}

		virtual void Perform(WorldManager* wm, 
							 ForceFieldManager* ffm, 
							 const MoveOverride& override) override
		{
			auto store = 0;

			// Get random world.
			World* w = wm->GetRandomWorld();

			Particle* plist[32];

			unsigned int NumberofParticles=1;

			// Unstash into particle list or single particle
			if(_multi_delete)
			{
				auto& comp = w->GetComposition();
				NumberofParticles = _species.size();
				for (unsigned int i = 0; i < _species.size(); i++)
				{
					if(comp[_species[i]] == 0)
						return;
					plist[i] = w->DrawRandomParticleBySpecies(_species[i]);
					if(plist[i] == nullptr) // Safety check.
						return;
				}
			}
			else
			{
				size_t n = _species.size();
				assert(n > 0);
				auto type = _rand.int32() % n;
				auto& comp = w->GetComposition();
				if(comp[type] == 0)
					return;
				plist[0] = w->DrawRandomParticleBySpecies(_species[type]);
				if(plist[0] == nullptr) // Safety check.
					return;
			}

			double Prefactor = 1;
			auto& sim = SimInfo::Instance();
			auto beta = 1.0/(sim.GetkB()*w->GetTemperature());



			// *************** NEW region based ADDITION ***********//
			//auto V = w->GetVolume();
			// even though evaluate energy and evaluate pressure takes into account world volume, since delta E and P, so it doesn't seem to matter
			// and delta E and P should accurately take into account delta E due to particle insertion 

			// NOTE:- THIS CURRENT IMPLEMENTATION IS FOR THE CONFINEMENT ONLY, MEANING WHEN THE ALL THE PARTICLES REMAIN THE CONFINEMENT SPECIFIED BY REGION INSERTION
			// AND THE PARTICLES DOES NOT EVER LEAVE THE CONFINED REGION

			// FOR SYSTEMS WHERE PARTICLE LEAVE THE REGION SPECIFIED BY REGION MOVE (E.G. MEMBRANE) THEN DIFFERENT IMPLEMENTATION REQUIRED. I.E. CURRENTLY NAMED AS CONFINEMENT
			// DELETION MOVE
			auto V = abs(_xhigh - _xlow) * abs(_yhigh - _ylow) * abs(_zhigh - _zlow);
			// ***************************************************//


			//EPTuple ei;
			auto& comp = w->GetComposition();
			
			// Get previous tail energy and pressure.
			auto wei = w->GetEnergy();
			auto wpi = w->GetPressure();

			for (unsigned int i = 0; i < NumberofParticles; i++)
			{

				auto id = plist[i]->GetSpeciesID();
				auto N = comp[id];




				// *************** NEW CONFINEMENT ADDITION ***********//
				auto mu = w->GetChemicalPotential(id);
				// you can use above statement but in that case the chemical pot set in the json script
				// is the chemical potential of species in the region

				// ***************************************************//






				auto lambda = w->GetWavelength(id);

				Prefactor*=(lambda*lambda*lambda*N)/V*exp(-beta*mu);
			}

			for (unsigned int i = 0; i < NumberofParticles; i++)
			{
				w->RemoveParticle(plist[i]);
			}
			auto ei = ffm->EvaluateEnergy(*w);
			++_performed;

			ei.energy -= wei; 
			ei.pressure -= wpi;

			// The acceptance rule is from Frenkel & Smit Eq. 5.6.9.
			//auto pacc = Prefactor*exp(beta*ei.energy.total()); // WRONG PACC CHECKED IT WITH GCMC LJ/ NIST TEST
			auto pacc = Prefactor*exp(-beta*ei.energy.total());
			pacc = pacc > 1.0 ? 1.0 : pacc;

			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				// Add it back to the world.
				for (unsigned int i = 0; i < NumberofParticles; i++)
				{
					w->AddParticle(plist[i]);
					// this adds the particle back to the same position
				}
				++_rejected;
			}
			else
			{
				// Stash the particle which actually removes it from the world. 
				for (unsigned int i = 0; i < NumberofParticles; i++)
				{
					w->StashParticle(plist[i]);
					store = i;
				}

				// Update energies and pressures.
				//w->IncrementEnergy(-1.0*ei.energy);
				//w->IncrementPressure(-1.0*ei.pressure);
				w->IncrementEnergy(ei.energy);
				w->IncrementPressure(ei.pressure);
				//std::cout << " THIS IS DELETE MOVE "<<std::endl;
				//std::cout << " PACC IS " << pacc <<std::endl;	
				//std::cout << " total energy is  " << ei.energy.total() <<std::endl;
				//std::cout << " id " << plist[store]->GetSpeciesID() <<std::endl;
				//std::cout << " Prefactor " << Prefactor <<std::endl;
				//std::cout << " ******************** "<<std::endl;
				//std::cout << "                       "<<std::endl;
			}
		}

		virtual void Perform(World* w, 
							 ForceFieldManager* ffm, 
							 DOSOrderParameter* op, 
							 const MoveOverride& override) override
		{

			Particle* plist[32];

			unsigned int NumberofParticles=1;

						// Unstash into particle list or single particle
			if(_multi_delete)
			{
				auto& comp = w->GetComposition();
				NumberofParticles=_species.size();
				for (unsigned int i = 0; i < _species.size(); i++)
				{
					if(comp[_species[i]] == 0)
						return;
					plist[i] = w->DrawRandomParticleBySpecies(_species[i]);
					if(plist[i] == nullptr) // Safety check.
						return;
				}
			}
			else
			{
				size_t n = _species.size();
				assert(n > 0);
				auto type = _rand.int32() % n;
				auto& comp = w->GetComposition();
				if(comp[type] == 0)
					return;
				plist[0] = w->DrawRandomParticleBySpecies(_species[type]);
				if(plist[0] == nullptr) // Safety check.
					return;
			}

			double Prefactor = 1;
			auto& sim = SimInfo::Instance();
			auto beta = 1.0/(sim.GetkB()*w->GetTemperature());
			auto V = w->GetVolume();
			auto& comp = w->GetComposition();
			
			auto opi = op->EvaluateOrderParameter(*w);
			EPTuple ei;

			for (unsigned int i = 0; i < NumberofParticles; i++)
			{
				auto id = plist[i]->GetSpeciesID();
				auto N = comp[id];
				auto mu = w->GetChemicalPotential(id);
				auto lambda = w->GetWavelength(id);

				if(_prefac)
					Prefactor*=(lambda*lambda*lambda*N)/V*exp(-beta*mu);

				// Evaluate old energy. For multi deletion moves
				// Need to remove particle one by one so energy
				// is not double counted.
				ei += ffm->EvaluateEnergy(*plist[i]);
				w->RemoveParticle(plist[i]);
			}

			++_performed;

			w->IncrementEnergy(-1.0*ei.energy);
			w->IncrementPressure(-1.0*ei.pressure);
			auto opf = op->EvaluateOrderParameter(*w);
			auto ef = w->GetEnergy();

			// Acceptance rule.
			double pacc = op->AcceptanceProbability(ei.energy, ef, opi, opf, *w);

			if(_prefac)
			{
				pacc *= Prefactor;
			}

			pacc = pacc > 1.0 ? 1.0 : pacc;

			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				// Add it back to the world.
				for (unsigned int i = 0; i < NumberofParticles; i++)
					w->AddParticle(plist[i]);

				w->IncrementEnergy(ei.energy);
				w->IncrementPressure(ei.pressure);
				++_rejected;
			}
			else
			{
				// Stash the particle which actually removes it from the world. 
				for (unsigned int i = 0; i < NumberofParticles; i++)
					w->StashParticle(plist[i]);
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
			json["type"] = "DeleteParticle";
			json["seed"] = _seed;
			json["op_prefactor"] = _prefac;
			json["multi_delte"] = _multi_delete;

			auto& species = Particle::GetSpeciesList();
			for(auto& s : _species)
				json["species"].append(species[s]);
		}

		virtual std::string GetName() const override { return "DeleteParticle"; }

		// Clone move.
		Move* Clone() const override
		{
			return new ConfinementDeleteParticleMove(static_cast<const ConfinementDeleteParticleMove&>(*this));
		}

		~ConfinementDeleteParticleMove(){}
	};
}