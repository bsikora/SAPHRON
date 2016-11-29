#pragma once 

#include "Move.h"
#include "lammps.h"
#include "input.h"
#include "atom.h"
#include "library.h"
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

using namespace LAMMPS_NS;

namespace SAPHRON
{
	class MDMove : public Move
	{
	private:

		MPI_Comm _comm_lammps;
		LAMMPS *_lmp;
		int _performed;
		int _rejected;
		bool _prefac;
		std::string _data_file;

		void ReadInputFile()
		{

		}

		//  WRITE THE LAMMPS DATA FILE
		void WriteDataFile(const World &world, LAMMPS *lmp)
		{
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

			int natoms = static_cast<int> (lmp->atom->natoms);
			int *image = new int[natoms];
			int *image_all = new int[3*natoms];
			int *type = new int[natoms];
			lammps_gather_atoms(lmp, "image", 0, 1, image);
			lammps_gather_atoms(lmp,"type", 0, 1, type);			

			world.GetPrimitiveCount();
			if(rank == 0)
			{
				std::ofstream datafile;
				datafile.open(_data_file,std::ofstream::out);
				datafile<<"Testing 123"<<std::endl;
				datafile.close();
			}

			delete [] image;
			delete [] image_all;
			delete [] type;
		}

	public:
		MDMove(std::string data_file) : 
		_comm_lammps(), _lmp(), _performed(0), _rejected(0), _prefac(0),
		_data_file(data_file)
		{
			MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &_comm_lammps);
		}

		virtual void Perform(WorldManager* wm, 
					 ForceFieldManager* ffm, 
					 const MoveOverride& override) override
		{
			World* w = wm->GetRandomWorld();
			WriteDataFile(*w, _lmp);

			//Write out data file

			//Create new lammps instance

			//Read in input file and run

			//Delete lammps instance
		}

		virtual void Perform(World*, 
							 ForceFieldManager*, 
							 DOSOrderParameter*, 
							 const MoveOverride&) override
		{
			std::cerr << "MD move does not support DOS interface." << std::endl;
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
		}

		virtual std::string GetName() const override { return "MD"; }

		// Clone move.
		Move* Clone() const override
		{
			return new MDMove(static_cast<const MDMove&>(*this));
		}

	};
}