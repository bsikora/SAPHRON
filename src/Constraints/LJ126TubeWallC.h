#pragma once 

#include "Constraint.h"
#include "../Particles/Particle.h"
#include "../Worlds/World.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include <armadillo>

namespace SAPHRON
{
	class LJ126TubeWallC : public Constraint
	{
	private:
			double _epsilon;
			double _sigma;
			double _rcut;
			double _cylrad;
			double _cylcenY;
			double _cylcenZ;
	public:
		LJ126TubeWallC(World* world, double epsilon, double sigma, double rcut, 
				double cylcen_y, double cylcen_z, double cylrad) : Constraint(world), _epsilon(epsilon), _sigma(sigma), _rcut(rcut), _cylrad(cylrad), 
			_cylcenY(cylcen_y), _cylcenZ(cylcen_z){}

		double EvaluateEnergy() const override
		{
			double LJ_E = 0;
			for(int i = 0; i < _world->GetPrimitiveCount(); ++i)
			{
				auto pi = _world->SelectPrimitive(i);
				Position ppos = pi->GetPosition();
				double dist_cyl = sqrt(pow(ppos[1]-_cylcenY, 2) + pow(ppos[2]-_cylcenZ, 2));
				double dist_wall = _cylrad - dist_cyl;
				if (dist_wall <= _rcut)
				{
					double LJ_E = 4.0*_epsilon*(pow(_sigma/dist_wall, 12) - pow(_sigma/dist_wall, 6)) - 
					4.0*_epsilon*(pow(_sigma/_rcut, 12) - pow(_sigma/_rcut, 6));
				}
			}
			return LJ_E;
		}

		void Serialize(Json::Value& json) const override
		{
			json["type"] = "LJ126TubeWall";
			json["epsilon"] = _epsilon;
			json["sigma"] = _sigma;
			json["rcut"] = _rcut;
			json["cylcen_y"] = _cylcenY;
			json["cylcen_z"] = _cylcenZ;
			json["cylrad"] = _cylrad;
		}
	};
}