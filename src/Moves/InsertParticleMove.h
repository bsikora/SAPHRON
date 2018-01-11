#pragma once

#include "Move.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	// Class for particle insertion move. Based on providing 
	// a list of species, the move will stash "stashcount" copies 
	// of each one, in each world in the world manager. However, 
	// if the number of stashed particles runs out (say, due to deletion)
	// then they are automatically replenished by the world, at an expense.
	class InsertParticleMove : public Move
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
		InsertParticleMove(const std::vector<int>& species, 
						   const WorldManager& wm,
						   int stashcount, bool multi_insert,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _scount(stashcount), 
		_multi_insert(multi_insert), _seed(seed)
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

		InsertParticleMove(const std::vector<std::string>& species, 
						   const WorldManager& wm,
						   int stashcount, bool multi_insert,
						   unsigned seed = 45843) :
		_rand(seed), _rejected(0), _performed(0), _species(0), 
		_prefac(true), _scount(stashcount), 
		_multi_insert(multi_insert), _seed(seed)
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
			auto V = w->GetVolume();
			auto& comp = w->GetComposition();

			
			// Get previous energy
			auto ei = ffm->EvaluateEnergy(*w);

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

				store_lambda = lambda;
				store_mu = mu;
				store_N = N;

				w->AddParticle(plist[i]);
				Prefactor*=V/(lambda*lambda*lambda*(N+1))*exp(beta*mu);
				store = i;

			}

			auto ef = ffm->EvaluateEnergy(*w);
			++_performed;

			auto de = ef - ei;

			// The acceptance rule is from Frenkel & Smit Eq. 5.6.8.
			auto pacc = Prefactor*exp(-beta*de.energy.total());
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
				//std::cout << " THIS IS INSERT MOVE "<<std::endl;
				//std::cout << " PACC IS: " << pacc <<std::endl;	
				//std::cout << " total energy is:  " << ef.energy.total() <<std::endl;
				//std::cout << " id: " << plist[store]->GetSpeciesID() <<std::endl;
				//std::cout << " Prefactor: " << Prefactor <<std::endl;
				//std::cout << " Volume: " << V <<std::endl;
				//std::cout << " lambda: " << store_lambda <<std::endl;
				//std::cout << " mu: " << store_mu <<std::endl;
				//std::cout << " N: " << store_N <<std::endl;
				//std::cout << " ******************** "<<std::endl;
				//std::cout << "                       "<<std::endl;
				/*
				std::cout << " *********** INSERT MOVE ACCEPTED!!!! ************** "<<std::endl;
				std::cout << "the GetEnergy method delta energy is "<<ef.energy.total()<<std::endl;
				std::cout << "the EvaluateEnergy method delta energy is "<<ef_store.energy.total()<<std::endl;
				std::cout << "the world energy after insert "<<w->GetEnergy().total()<<std::endl;
				std::cout << " *********** IIIIIIIIIIIIIIIIIIIIIIIII ************** "<<std::endl;
				std::cout << "                                                      "<<std::endl;
				*/


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
			auto V = w->GetVolume();
			auto& comp = w->GetComposition();

			
			// Get previous energy
			auto ei = ffm->EvaluateEnergy(*w);
			auto opi = op->EvaluateOrderParameter(*w);
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


				w->AddParticle(plist[i]);
				if(_prefac)
					Prefactor*=V/(lambda*lambda*lambda*(N+1))*exp(beta*mu);
			
			}

			auto ef = ffm->EvaluateEnergy(*w);
			auto opf = op->EvaluateOrderParameter(*w);
			++_performed;


			// The acceptance rule is from Frenkel & Smit Eq. 5.6.8.
			// However, it iwas modified since we are using the *final* particle number.
			double pacc = op->AcceptanceProbability(ei.energy, ef.energy, opi, opf, *w);
			
			// If prefactor is enabled, compute.
			if(_prefac)
				pacc *= Prefactor;

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
			return new InsertParticleMove(static_cast<const InsertParticleMove&>(*this));
		}

		~InsertParticleMove(){}
	};
}