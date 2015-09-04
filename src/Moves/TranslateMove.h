#pragma once 

#include "../Rand.h"
#include "Move.h"
#include "../Worlds/WorldManager.h"
#include "../Simulation/SimInfo.h"
#include "../ForceFields/ForceFieldManager.h"

namespace SAPHRON
{
	// Class for translating a random particle a maximum of "dx" distance.
	class TranslateMove : public Move
	{
		private: 
			double _dx;
			Rand _rand;
			int _rejected;
			int _performed;
			int _seed;

		public: 
			TranslateMove(double dx, int seed = 2496) : 
			_dx(dx), _rand(seed), _rejected(0), _performed(0), _seed(seed)
			{
			}

			// This is for unit testing convenience. It is not part of the interface.
			void Perform(Particle* particle)
			{
				auto& pos = particle->GetPositionRef();
				particle->SetPosition(pos.x + _dx*(_rand.doub()-0.5), 
									   pos.y + _dx*(_rand.doub()-0.5), 
									   pos.z + _dx*(_rand.doub()-0.5));
				++_performed;
			}

			// Perform translation on a random particle from a random world.
			virtual void Perform(WorldManager* wm, ForceFieldManager* ffm, const MoveOverride& override) override
			{
				// Get random particle from random world.
				World* w = wm->GetRandomWorld();
				Particle* particle = w->DrawRandomParticle();

				// Initial position.
				const Position& posi = particle->GetPositionRef();
				
				// Evaluate initial particle energy. 
				auto ei = ffm->EvaluateHamiltonian(*particle, w->GetComposition(), w->GetVolume());

				// Generate new position then apply periodic boundaries.
				Position newPos({posi.x + _dx*(_rand.doub()-0.5), 
								 posi.y + _dx*(_rand.doub()-0.5), 
								 posi.z + _dx*(_rand.doub()-0.5)});
				
				w->ApplyPeriodicBoundaries(&newPos);
				particle->SetPosition(newPos);
				++_performed;										

				// Evaluate final particle energy and get delta E. 
				auto ef = ffm->EvaluateHamiltonian(*particle, w->GetComposition(), w->GetVolume());
				Energy de = ef.energy - ei.energy;
				
				// Get sim info for kB.
				auto sim = SimInfo::Instance();

				// Acceptance probability.
				double p = exp(-de.total()/(w->GetTemperature()*sim.GetkB()));
				p = p > 1.0 ? 1.0 : p;

				// Reject or accept move.
				if(!(override == ForceAccept) && (p < _rand.doub() || override == ForceReject))
				{
					particle->SetPosition(posi);
					++_rejected;
				}
				else
				{
					// Update energies and pressures.
					w->SetEnergy(w->GetEnergy() + de);
					w->SetPressure(w->GetPressure() + (ef.pressure - ei.pressure));
				}	
			}
		
			// Returns maximum displacement.
			double GetMaxDisplacement()
			{
				return _dx;
			}

			virtual double GetAcceptanceRatio() const override
			{
				return 1.0-(double)_rejected/_performed;
			};

			virtual void ResetAcceptanceRatio() override
			{
				_performed = 0;
				_rejected = 0;
			}

			// Get seed.
			virtual int GetSeed() const override { return _seed; }

			virtual std::string GetName() const override { return "Translate"; }

			// Clone move.
			Move* Clone() const override
			{
				return new TranslateMove(static_cast<const TranslateMove&>(*this));
			}

	};
}