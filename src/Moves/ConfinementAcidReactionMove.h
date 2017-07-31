#pragma once 

#include "Move.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	class ConfinementAcidReactionMove : public Move
	{
	private:
		std::vector<int> _swap;
		std::vector<int> _products;
		Rand _rand;

		int _performed;
		int _rejected;
		bool _prefac;
		int _scount;
		double _mu;

		double _m1;
		double _m2;

		int _i1;
		int _i2;

		double _c1;
		double _c2;

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
			for(auto& id : _products)
			{
				// Since the species exists, we assume there must be at least 
				// one find. 
				auto pcand = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);

				auto* P=pcand->second->Clone();
				// Stash a characteristic amount of the particles in world.
				for(auto& world : wm)
					world->StashParticle(P, _scount);
			}

		}

	public:
		ConfinementAcidReactionMove(const std::vector<std::string>& swap,
		const std::vector<std::string>& products,
		const WorldManager& wm,
		int stashcount, double mu,
		double xlow, double ylow, double zlow,
		double xhigh, double yhigh, double zhigh,
		unsigned seed = 7456253) : 
		_swap(0),_products(0), _rand(seed), _performed(0),
		 _rejected(0), _prefac(true), _scount(stashcount),
		 _mu(mu), _m1(0), _m2(0), _i1(0), _i2(1), _c1(0),
		 _c2(0), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
		{
			// Verify species list and add to local vector.
			auto& list = Particle::GetSpeciesList();

			for(auto& id : swap)
			{
				auto it = std::find(list.begin(), list.end(), id);
				if(it == list.end())
				{
					std::cerr << "Species ID \""
							  << id << "\" provided does not exist."
							  << std::endl;
					exit(-1);
				}
				_swap.push_back(it - list.begin());
			}

			for(auto& id : products)
			{
				auto it = std::find(list.begin(), list.end(), id);
				if(it == list.end())
				{
					std::cerr << "Species ID \""
							  << id << "\" provided does not exist."
							  << std::endl;
					exit(-1);
				}
				_products.push_back(it - list.begin());
			}

			InitStashParticles(wm);

			auto& plist = Particle::GetParticleMap();

			auto id =_swap[0];
			auto p1 = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);
			id =_swap[1];			
			auto p2 = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);
			
			_c1=p1->second->GetCharge();
			_c2=p2->second->GetCharge();

			_m1=p1->second->GetMass();
			_m2=p2->second->GetMass();

			_i1 = p1->second->GetSpeciesID();
			_i2 = p2->second->GetSpeciesID();

		}

		ConfinementAcidReactionMove(const std::vector<int>& swap,
		const std::vector<int>& products,
		const WorldManager& wm,
		int stashcount, double mu, 
		double xlow, double ylow, double zlow,
		double xhigh, double yhigh, double zhigh,
		unsigned seed = 7456253) : 
		_swap(0),_products(0), _rand(seed), _performed(0),
		 _rejected(0), _prefac(true), _scount(stashcount),
		 _mu(mu), _m1(0), _m2(0), _i1(0), _i2(1), _c1(0),
		 _c2(0), _xlow(xlow), _ylow(ylow), _zlow(zlow), _xhigh(xhigh), _yhigh(yhigh), _zhigh(zhigh), _seed(seed)
		{
			// Verify species list and add to local vector.
			auto& list = Particle::GetSpeciesList();
			
			for(auto& id : swap)
			{
				if(id >= (int)list.size())
				{
					std::cerr << "Species ID \"" 
							  << id << "\" provided does not exist." 
							  << std::endl;
					exit(-1);
				}
				_swap.push_back(id);
			}
			
			for(auto& id : products)
			{
				if(id >= (int)list.size())
				{
					std::cerr << "Species ID \"" 
							  << id << "\" provided does not exist." 
							  << std::endl;
					exit(-1);
				}
				_products.push_back(id);
			}

			InitStashParticles(wm);

			auto& plist = Particle::GetParticleMap();

			auto id =_swap[0];
			auto p1 = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);
			id =_swap[1];			
			auto p2 = std::find_if(plist.begin(), plist.end(), 
					[=](const std::pair<int, Particle*>& p)
					{
						return p.second->GetSpeciesID() == id;
					}
				);
			
			_c1=p1->second->GetCharge();
			_c2=p2->second->GetCharge();

			_m1=p1->second->GetMass();
			_m2=p2->second->GetMass();

			_i1 = p1->second->GetSpeciesID();
			_i2 = p2->second->GetSpeciesID();
		} 



		// wall energy calculation
		double ConfinedWallEnergy(Particle* pi)
		{
			double epsilon = 1.0;
			double sigma = 1.0;
			double wall_rc = 1.122;
			double LJ_wall_E = 0.0;
			double LJ_E = 0;


			double r_y_high = fabs(pi->GetPosition()[1] - _yhigh);
			double r_y_low = fabs(pi->GetPosition()[1] - _ylow);
			double r_z_high = fabs(pi->GetPosition()[2] - _zhigh);
			double r_z_low = fabs(pi->GetPosition()[2] - _zlow);

			std::cout <<"THE yhigh is  " << r_y_high <<std::endl; //*********
			std::cout <<"THE zlow is  " << r_z_low <<std::endl; //*********

			double arr[] = {r_y_high, r_y_low, r_z_high, r_z_low};
			std::vector<double> vec (arr, arr + sizeof(arr) / sizeof(arr[0]) );
			
			for (std::vector<int>::size_type k = 0; k != vec.size(); k++)
			{
				double r_calc = vec[k];
				
				if (r_calc <= wall_rc)
				{
					LJ_E = 4.0*epsilon*(pow(sigma/r_calc, 12) - pow(sigma/r_calc,6)) - 
					4.0*epsilon*(pow(sigma/wall_rc, 12) - pow(sigma/wall_rc,6)); 
					std::cout << std::fixed <<"THE LJ_E  " << LJ_E <<std::endl;
					std::cout << std::fixed <<"I AM BELOW OR AT WALL_RC  " << r_calc <<std::endl;

				}else
				{
					LJ_E = 0.0;
				}

				LJ_wall_E += LJ_E;
			}

			std::cout << std::fixed <<"LJ_wall_E is  " << LJ_wall_E <<std::endl; 

			// !!!! THE PACC NOW WILL BE, UNCOMMENT THE BELOW PACC EQUATION AND REPLACE WITH THE ACTUAL PACC BELOW (AT OR AROUND LINE 351) !!!!
			//auto pacc = Prefactor*exp(-beta*ef.energy.total()-LJ_wall_E); // PACC WITH WALL ENERGY ADDED IN

			return LJ_wall_E;
			// ***********WALL INTERACTION ENERGY ADDED**************************
		}



		virtual void Perform(WorldManager* wm, 
					 ForceFieldManager* ffm, 
					 const MoveOverride& override) override
		{

			// Draw random particle from random world.
			auto* w = wm->GetRandomWorld();
			Particle* p1 = nullptr;
			Particle* p2 = nullptr;
			Particle* ph = nullptr;

			int RxnExtent = 1;
			double rxndirection = _rand.doub();
			double bias = 1.0;
			auto& comp = w->GetComposition();

			//Determine reaction direction.
			if(rxndirection>=0.5) //Forward reaction
			{
				RxnExtent = 1;
				p1=w->DrawRandomPrimitiveBySpecies(_swap[0]);

				if(p1==nullptr)
					return;

				bias = double(comp[_swap[0]])/(comp[_swap[0]]+comp[_swap[1]]);
			}

			else //Reverse reaction
			{
				RxnExtent = -1;
				p2 = w->DrawRandomPrimitiveBySpecies(_swap[1]);
				ph = w->DrawRandomPrimitiveBySpecies(_products[0]);
	
				if(ph==nullptr && p2==nullptr)
					return;
				
				else if(ph==nullptr || p2==nullptr)
				{
					////std::cout<<"Only Half of the products present! Exiting"<<std::endl;
					//exit(-1);
					return;
				}

				bias = double(comp[_swap[1]])/(comp[_swap[0]]+comp[_swap[1]]);
			}

			double lambdaratio;

			int comp1 = comp[_i1];
			int comp2 = comp[_i2];
			int compph = comp[_products[0]];



			// *************** NEW region based ADDITION ***********//

			//auto V = pow(w->GetVolume(),RxnExtent); 
			// get the specified volume region
			auto V = pow(abs(_xhigh - _xlow) * abs(_yhigh - _ylow) * abs(_zhigh - _zlow), RxnExtent);
			double WALL_E = 0;

			// ***************************************************//
			


			auto lambda = w->GetWavelength(_products[0]);
			auto lambda3 = pow(lambda*lambda*lambda,-1*RxnExtent);
			double Nratio=0;
			double Korxn=0;

			// Evaluate initial energy. 
			EPTuple ei, ef;

			if(RxnExtent == -1)
			{
				Nratio = comp2*compph/(comp1 + 1.0);
				Korxn = exp(_mu);
				//**ei = ffm->EvaluateEnergy(*ph);
				ei = ffm->EvaluateEnergy(*w);
				w->RemoveParticle(ph);
				//**ei += ffm->EvaluateEnergy(*p2);
				p2->SetCharge(_c1);
				p2->SetMass(_m1);
				p2->SetSpeciesID(_i1);

				//**ef = ffm->EvaluateEnergy(*p2);
				ef = ffm->EvaluateEnergy(*w);

				/***NEW wall interaction energy***/
				WALL_E = 0;
				/*** NEW wall interaction energy***/

				lambdaratio = pow(_m1/_m2,3.0/2.0);
			}

			else
			{
				Nratio = comp1/((comp2+1.0)*(compph+1.0));
				Korxn = exp(-_mu);

				//**ei = ffm->EvaluateEnergy(*p1);
				ei = ffm->EvaluateEnergy(*w);

				p1->SetCharge(_c2);
				p1->SetMass(_m2);
				p1->SetSpeciesID(_i2);
				
				//**ef = ffm->EvaluateEnergy(*p1);
				
				ph = w->UnstashParticle(_products[0]);
				// Generate a random position and orientation for particle insertion.
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





				ph->SetPosition(pos);

				// Insert particle.
				w->AddParticle(ph);


				/*** NEW wall interaction energy***/
				double WALL_E = ConfinedWallEnergy(ph);
				std::cout << std::fixed <<"wall energy from separate method  " << WALL_E <<std::endl;
				/*** NEW wall interaction energy***/


				//**ef += ffm->EvaluateEnergy(*ph);
				ef = ffm->EvaluateEnergy(*w);

				lambdaratio = pow(_m2/_m1,3.0/2.0);
			}

			++_performed;

			auto de = ef - ei;
		
			// Get sim info for kB.
			auto& sim = SimInfo::Instance();

			// Acceptance probability.
			// removed bias
			double pacc = Nratio*V*lambda3*lambdaratio*Korxn*
			exp((-de.energy.total())/(w->GetTemperature()*sim.GetkB()));


			/*** NEW wall interaction energy***/
			//double pacc = Nratio*V*lambda3*lambdaratio*Korxn*exp((-de.energy.total()-WALL_E)/(w->GetTemperature()*sim.GetkB()));
			/*** NEW wall interaction energy***/


			pacc = pacc > 1.0 ? 1.0 : pacc;

			// Reject or accept move.
			if(!(override == ForceAccept) && (pacc < _rand.doub() || override == ForceReject))
			{
				++_rejected;
				if(RxnExtent == -1) 
				{
					p2->SetCharge(_c2);
					p2->SetMass(_m2);
					p2->SetSpeciesID(_i2);
					w->AddParticle(ph);   
					// ***** NOT SURE IF THIS ADDS PARTICLE AT THE SAME PLACE IT WAS REMOVED FROM WHEN MOVE WAS ATTEMPTED ????
					// YES IT ADDS IT BACK IN THE SAME PLACE
				}

				else 
				{
					p1->SetCharge(_c1);
					p1->SetMass(_m1);
					p1->SetSpeciesID(_i1);
					w->RemoveParticle(ph);
					w->StashParticle(ph);
				}

			}

			else
			{
				if(RxnExtent == -1)
				{
					w->StashParticle(ph);
				}

				w->IncrementEnergy(de.energy);
				w->IncrementPressure(de.pressure);
			}

			//std::cout << " PACC IS " << pacc <<std::endl;
			//std::cout << " rxndirection is " << rxndirection <<std::endl;
			//std::cout << " RxnExtent is " << RxnExtent <<std::endl;
			//std::cout << " lambda ratio is " << lambdaratio <<std::endl;		
			//std::cout << " total energy is  " << de.energy.total() <<std::endl;
			//std::cout << " Nratio is " << Nratio <<std::endl;
			//std::cout << " Korxn is  " << Korxn <<std::endl;
			//std::cout << " swap[0] is " << _swap[0] <<std::endl;
			//std::cout << " swap[1] is " << _swap[1] <<std::endl;
			//std::cout << " products[0] is " << _products[0] <<std::endl;
			//std::cout << " comp1 is " << comp1 <<std::endl;
			//std::cout << " comp2 is " << comp2 <<std::endl;
			//std::cout << " compph is " << compph <<std::endl;
			//std::cout << " ******************** " << compph <<std::endl;
			//std::cout << " ******************** " << compph <<std::endl;


		}

		virtual void Perform(World*, 
							 ForceFieldManager*, 
							 DOSOrderParameter*, 
							 const MoveOverride&) override
		{
			std::cerr << "Acid Reaction move does not support DOS interface." << std::endl;
			exit(-1);
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
			json["type"] = GetName();
			json["seed"] = _seed;
			json["mu"] = _mu;
			json["stash_count"] = _scount;
			json["op_prefactor"] = _prefac;

			auto& slist = Particle::GetSpeciesList();
			for(auto& s : _swap)
				json["swap"].append(slist[s]);
			for(auto& s : _products)
				json["products"].append(slist[s]);
		}

		virtual std::string GetName() const override { return "AcidReaction"; }

		// Clone move.
		Move* Clone() const override
		{
			return new ConfinementAcidReactionMove(static_cast<const ConfinementAcidReactionMove&>(*this));
		}

	};
}