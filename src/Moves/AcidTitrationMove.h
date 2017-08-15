#pragma once 
#include "Move.h"
#include "../Particles/Particle.h"
#include "../Worlds/WorldManager.h"
#include "../Worlds/World.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../Utils/Rand.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	class AcidTitrationMove : public Move
	{
	private:
		std::vector<int> _species;
		Rand _rand;

		int _performed;
		int _rejected;
		unsigned _seed;

		double _protoncharge;
		double _mu;

		bool _prefac;

	public:
		AcidTitrationMove(const std::vector<std::string>& species,
		double protoncharge, double mu, unsigned seed = 7456253) : 
		_species(0), _rand(seed), _performed(0), _rejected(0),
		_seed(seed), _protoncharge(protoncharge), _mu(mu),
		_prefac(true)
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

		AcidTitrationMove(const std::vector<int>& species,
		double protoncharge, double mu, unsigned seed = 7456253) : 
		_species(0), _rand(seed), _performed(0), _rejected(0),
		_seed(seed), _protoncharge(protoncharge), _mu(mu),
		_prefac(true)
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

		virtual void Perform(WorldManager* wm, 
					 ForceFieldManager* ffm, 
					 const MoveOverride& override) override
		{
			// Draw random particle from random world.
			auto* w = wm->GetRandomWorld();
			auto* p = w->DrawRandomPrimitive();

			if(p == nullptr)
				return;

			if(std::find(_species.begin(), _species.end(), p->GetSpeciesID()) == _species.end())
				return;

			// Evaluate initial energy. 
			auto ei = ffm->EvaluateEnergy(*p);

			// Perform protonation/deprotonation.
			auto tc = p->GetCharge();
			auto amu = _mu;
			auto randomnumber = _rand.doub();

			//Deprotonate acid if protonated
			if(randomnumber > 0.50 && tc==0.0)
			{
				p->SetCharge(-_protoncharge);
				amu = -_mu;
			}


			//Protonate if deprotonated
			else if(randomnumber <= 0.50 && tc != 0.0)
			{
				p->SetCharge(0.0);
			}
			else
				return;

			++_performed;

			auto ef = ffm->EvaluateEnergy(*p);
			auto de = ef - ei;
		
			// Get sim info for kB.
			auto& sim = SimInfo::Instance();

			// Acceptance probability.
			double pacc = exp((-de.energy.total()+amu)/(w->GetTemperature()*sim.GetkB()));
			pacc = pacc > 1.0 ? 1.0 : pacc;

			// Reject or accept move.
			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				p->SetCharge(tc);
				++_rejected;
			}
			else
			{
				// Update energies and pressures.
				w->IncrementEnergy(de.energy);
				w->IncrementPressure(de.pressure);

				/********************************/
				// Update energies and pressures.
				//auto wor_ef = ffm->EvaluateEnergy(*w);
				//w->SetEnergy(wor_ef.energy);
				//w->SetPressure(wor_ef.pressure);
				/********************************/
			}
		}

		// Perform move interface for DOS ensemble.
		virtual void Perform(World* world, 
							 ForceFieldManager* ffm, 
							 DOSOrderParameter* op, 
							 const MoveOverride& override) override
		{

			auto* p = world->DrawRandomPrimitive();

			if(p == nullptr)
				return;

			if(std::find(_species.begin(), _species.end(), p->GetSpeciesID()) == _species.end())
				return;

			// Evaluate initial energy. 
			auto ei = ffm->EvaluateEnergy(*p);
			auto opi = op->EvaluateOrderParameter(*world);

			// Perform protonation/deprotonation.
			auto tc=p->GetCharge();
			auto amu = _mu;
			auto randomnumber = _rand.doub();

			//Deprotonate acid if protonated
			if(randomnumber > 0.50 && tc==0.0)
			{
				p->SetCharge(-_protoncharge);
				amu = -_mu;
			}

			//Protonate if deprotonated
			else if(randomnumber <= 0.50 && tc != 0.0)
			{
				p->SetCharge(0.0);
			}
			else
				return;
			
			++_performed;

			auto ef = ffm->EvaluateEnergy(*p);
			auto de = ef - ei;

			// Update energies and pressures.
			world->IncrementEnergy(de.energy);
			world->IncrementPressure(de.pressure);

			auto opf = op->EvaluateOrderParameter(*world);
			
			// Acceptance probability.
			double pacc = op->AcceptanceProbability(ei.energy, ef.energy, opi, opf, *world);
			if(_prefac)
			{
				auto& sim = SimInfo::Instance();
				auto beta = 1.0/(world->GetTemperature()*sim.GetkB());
				auto arg = beta*(amu);
				pacc *= exp(arg);
			}
			pacc = pacc > 1.0 ? 1.0 : pacc;

			// Reject or accept move.
			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				p->SetCharge(tc);
				world->IncrementEnergy(-1.0*de.energy);
				world->IncrementPressure(-1.0*de.pressure);
				++_rejected;
			}
		}

		// Turns on or off the acceptance rule prefactor for DOS order parameter.
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
			auto& refspecies = Particle::GetSpeciesList();

			json["type"] = GetName();
			json["seed"] = _seed;
			json["mu"] = _mu;
			json["proton_charge"] = _protoncharge;
			json["op_prefactor"] = _prefac;

			for(auto& s : _species)
				json["species"].append(refspecies[s]);
		}

		virtual std::string GetName() const override { return "AcidTitrate"; }

		// Clone move.
		Move* Clone() const override
		{
			return new AcidTitrationMove(static_cast<const AcidTitrationMove&>(*this));
		}

	};
}