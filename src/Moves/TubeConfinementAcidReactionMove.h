#pragma once 

#include "Move.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

namespace SAPHRON
{
	class TubeConfinementAcidReactionMove : public Move
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
		double _cylrad;
		double _cylcenY;
		double _cylcenZ;

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
		TubeConfinementAcidReactionMove(const std::vector<std::string>& swap,
		const std::vector<std::string>& products,
		const WorldManager& wm,
		int stashcount, double mu,double cylrad, double cylcen_y, double cylcen_z,unsigned seed = 7456253) : 
		_swap(0),_products(0), _rand(seed), _performed(0),
		 _rejected(0), _prefac(true), _scount(stashcount),
		 _mu(mu), _m1(0), _m2(0), _i1(0), _i2(1), _c1(0),
		 _c2(0), _cylrad(cylrad), _cylcenY(cylcen_y), _cylcenZ(cylcen_z), _seed(seed)
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

		TubeConfinementAcidReactionMove(const std::vector<int>& swap,
		const std::vector<int>& products,
		const WorldManager& wm,
		int stashcount, double mu, double cylrad, double cylcen_y, double cylcen_z,unsigned seed = 7456253) : 
		_swap(0),_products(0), _rand(seed), _performed(0),
		 _rejected(0), _prefac(true), _scount(stashcount),
		 _mu(mu), _m1(0), _m2(0), _i1(0), _i2(1), _c1(0),
		 _c2(0), _cylrad(cylrad), _cylcenY(cylcen_y), _cylcenZ(cylcen_z), _seed(seed)
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
			//double bias = 1.0;
			auto& comp = w->GetComposition();

			//Determine reaction direction.
			if(rxndirection>=0.5) //Forward reaction
			{
				RxnExtent = 1;
				p1=w->DrawRandomPrimitiveBySpecies(_swap[0]);

				if(p1==nullptr)
					return;

				//bias = double(comp[_swap[0]])/(comp[_swap[0]]+comp[_swap[1]]);
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

				//bias = double(comp[_swap[1]])/(comp[_swap[0]]+comp[_swap[1]]);
			}

			double lambdaratio;

			int comp1 = comp[_i1];
			int comp2 = comp[_i2];
			int compph = comp[_products[0]];

			//auto V = pow(w->GetVolume(),RxnExtent);
			const auto& H = w->GetHMatrix();
			auto V = pow(M_PI*_cylrad*_cylrad*H(0,0),RxnExtent);
			auto lambda = w->GetWavelength(_products[0]);
			auto lambda3 = pow(lambda*lambda*lambda,-1*RxnExtent);
			double Nratio=0;
			double Korxn=0;

			EPTuple ei, ef;

			if(RxnExtent == -1)
			{
				Nratio = comp2*compph/(comp1 + 1.0);
				Korxn = exp(_mu);

				ei = ffm->EvaluateEnergy(*w);

				w->RemoveParticle(ph);
				p2->SetCharge(_c1);
				p2->SetMass(_m1);
				p2->SetSpeciesID(_i1);
				ef = ffm->EvaluateEnergy(*w);
				lambdaratio = pow(_m1/_m2,3.0/2.0);
			}

			else
			{
				Nratio = comp1/((comp2+1.0)*(compph+1.0));
				Korxn = exp(-_mu);

				ei = ffm->EvaluateEnergy(*w);

				p1->SetCharge(_c2);
				p1->SetMass(_m2);
				p1->SetSpeciesID(_i2);				
				ph = w->UnstashParticle(_products[0]);
				// Generate a random position and orientation for particle insertion.
				//const auto& H = w->GetHMatrix();
				//Vector3D pr{_rand.doub(), _rand.doub(), _rand.doub()};
				//Vector3D pos = H*pr;

				/*!!!!ADDITOINAL TUBE CONFINEMENT !!!!***/
				double theta = _rand.doub()*2*M_PI;
				double r_cyl = _rand.doub()*_cylrad; //(input)
				double x_cyl = _rand.doub()*H(0,0); // cause x axis is my cylinderical axis
				double y_cyl = sqrt(r_cyl)*cos(theta);
				double z_cyl = sqrt(r_cyl)*sin(theta);
				Vector3D pos{x_cyl, (y_cyl+_cylcenY), (z_cyl+_cylcenZ)};
				/*!!!!ADDITOINAL TUBE CONFINEMENT !!!!***/

				ph->SetPosition(pos);
				// Insert particle.
				w->AddParticle(ph);
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

				// Update energies and pressures.
				w->SetEnergy(ef.energy);
				w->SetPressure(ef.pressure);
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
			//std::cout << " postion of OH is " << ph->GetPosition() <<std::endl;
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
			json["cylrad"] = _cylrad;
			json["cylcen_y"] = _cylcenY;
			json["cylcen_z"] = _cylcenZ;

			auto& slist = Particle::GetSpeciesList();
			for(auto& s : _swap)
				json["swap"].append(slist[s]);
			for(auto& s : _products)
				json["products"].append(slist[s]);
		}

		virtual std::string GetName() const override { return "TubeConfinementAcidReaction"; }

		// Clone move.
		Move* Clone() const override
		{
			return new TubeConfinementAcidReactionMove(static_cast<const TubeConfinementAcidReactionMove&>(*this));
		}

	};
}