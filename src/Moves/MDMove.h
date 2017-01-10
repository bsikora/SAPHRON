#pragma once 

#include "Move.h"
#include <map>
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

// This move currently only supports **one** type of bond
namespace SAPHRON
{
	class MDMove : public Move
	{
	private:

		Rand _rand;
		MPI_Comm _comm_lammps;
		LAMMPS *_lmp;
		std::string _data_file;
		std::string _input_file;
		std::string _minimize_file;
		std::map<int, int> _S2L_map;

		std::map<int, double> _S2L_vxmap; ///*****
		std::map<int, double> _S2L_vymap; ///*****
		std::map<int, double> _S2L_vzmap; ///*****

		std::map<std::string, int> _S2L_imap;
		std::map<int, Particle*> _L2S_map;

		std::map<Particle*, double> _L2S_vxmap; ///*****
		std::map<Particle*, double> _L2S_vymap; ///*****
		std::map<Particle*, double> _L2S_vzmap; ///*****


		int _bondnumber;
		int _atomnumber;

		// matches spahron id to lammps ids
		void UpdateMap(const World &world)
		{
			int lammps_id = 1;
			for(auto& p : world)
			{
				if(p->HasChildren())
				{
					for(auto& cp : *p)
					{
						_S2L_map[cp->GetGlobalIdentifier()] = lammps_id;
						_S2L_vxmap[cp->GetGlobalIdentifier()] = _L2S_vxmap[cp]; ///*****
						_S2L_vymap[cp->GetGlobalIdentifier()] = _L2S_vymap[cp]; ///*****
						_S2L_vzmap[cp->GetGlobalIdentifier()] = _L2S_vzmap[cp]; ///*****
						lammps_id++;
					}
				}
				else
				{
					_S2L_map[p->GetGlobalIdentifier()] = lammps_id;
					_S2L_vxmap[p->GetGlobalIdentifier()] = _L2S_vxmap[p]; ///*****
					_S2L_vymap[p->GetGlobalIdentifier()] = _L2S_vymap[p]; ///*****
					_S2L_vzmap[p->GetGlobalIdentifier()] = _L2S_vzmap[p]; ///*****
					lammps_id++;
				}
			}
		}

		void AnalyzeParticle(Particle* p, std::string& coords, std::string& bonding, std::string& velocities)
		{
			_L2S_map[_atomnumber] = p;
			_atomnumber++;

			auto pid = _S2L_map[p->GetGlobalIdentifier()];
			auto sid = _S2L_imap[p->GetSpecies()];

			auto vx = _S2L_vxmap[p->GetGlobalIdentifier()]; ///*****
			auto vy = _S2L_vymap[p->GetGlobalIdentifier()]; ///*****
			auto vz = _S2L_vzmap[p->GetGlobalIdentifier()]; ///*****

			Position ppos = p->GetPosition();

			coords += std::to_string(pid) + " 1 " + 
					std::to_string(sid) + " " +
					std::to_string(p->GetCharge()) + " " +
					std::to_string(ppos[0]) + " " +
					std::to_string(ppos[1]) + " " +
					std::to_string(ppos[2]) + " 0 0 0\n";

			velocities += std::to_string(pid) + " " + ///*****
					std::to_string(vx) + " " + ///*****
					std::to_string(vy) + " " + ///*****
					std::to_string(vz) + "\n"; ///*****

			NeighborList bondedneighbors = p->GetBondedNeighbors();
			if(bondedneighbors.size() == 0)
				return;

			for(auto& bnp : bondedneighbors)
			{
				auto bnpid = _S2L_map[bnp->GetGlobalIdentifier()];
				if(bnpid > pid)
				{
					_bondnumber++;
					bonding += std::to_string(_bondnumber) + " 1 " + 
					std::to_string(pid) + 
					" " + std::to_string(bnpid) +"\n";
				}
			}
		}

		//  WRITE THE LAMMPS DATA FILE
		void WriteDataFile(const World &world)
		{
			
			//Determine bonding
			std::string bonding = "Bonds\n\n";
			std::string coords = "Atoms\n\n";
			std::string velocities = "Velocities\n\n"; ///*****
			
			_bondnumber = 0;
			_atomnumber = 0;

			for(auto& p : world)
				if(p->HasChildren())
					for(auto& cp : *p)
						AnalyzeParticle(cp, coords, bonding, velocities); ///*****
				else
					AnalyzeParticle(p, coords, bonding, velocities); ///*****

			if(_bondnumber == 0)
				bonding = "";

			std::ofstream datafile;
			datafile.open(_data_file,std::ofstream::out);
			
			datafile<<"LAMMPS Testing file \n\n";
			datafile<<std::to_string(_atomnumber) + " atoms\n";
			datafile<<std::to_string(_bondnumber) + " bonds\n";
			datafile<<"0 angles\n";
			datafile<<"0 dihedrals\n";
			datafile<<"0 impropers\n\n";

			datafile<<std::to_string(world.GetComposition().size()) + " atom types\n";
			datafile<<"1 bond types\n";
			datafile<<"0 angle types\n";
			datafile<<"0 dihedral types\n";
			datafile<<"0 improper types\n\n";

			auto& box = world.GetHMatrix();
			datafile<<"0 "+std::to_string(box(0,0)) +" xlo xhi\n";
			datafile<<"0 "+std::to_string(box(1,1)) +" ylo yhi\n";
			datafile<<"0 "+std::to_string(box(2,2)) +" zlo zhi\n\n";

			datafile<<"Masses\n\n";
			for(int i = 0; i < world.GetComposition().size(); i++)
				datafile<<std::to_string(i+1) + " 1.0\n";
			datafile<<std::endl;
			datafile<<coords;
			datafile<<velocities; ///*****
			datafile<<bonding;

			datafile.close();
		}

		// Update SAPHRON from LAMMPS
		void ReadInputFile(std::string file_to_read)
		{
			// open LAMMPS input script
			FILE *fp;
			int rank;
			MPI_Comm_rank(_comm_lammps, &rank);
			if (rank == 0)
			{
				fp = fopen(file_to_read.c_str(),"r");
				if (fp == NULL)
				{
					throw std::logic_error("ERROR: Could not open LAMMPS input script " + file_to_read);
				}
			}

			// Read lammps file line by line
			int n;
			char line[1024];
			while (1) 
			{
				if (rank == 0) 
				{
					if (fgets(line,1024,fp) == NULL) n = 0;
					else n = strlen(line) + 1;
					if (n == 0) fclose(fp);
				}

				MPI_Bcast(&n,1,MPI_INT,0,_comm_lammps);
				if (n == 0) break;
					MPI_Bcast(line,n,MPI_CHAR,0,_comm_lammps);

				std::string test = line;

				size_t f = test.find("RANDOM");
				if (f != std::string::npos)
				{
					test.replace(f, std::string("RANDOM").length(), std::to_string(int(_rand.int32()/1000)));
					_lmp->input->one(test.c_str());
				}
				else
				{
					_lmp->input->one(line);
				}
			}
		}

		void UpdateSAPHRON(const World &world)
		{
			int natoms = static_cast<int> (_lmp->atom->natoms);

			if(world.GetPrimitiveCount() != natoms)
			{
				std::cout<<"LAMMPS N = "<<natoms<<". SAPHRON N = "<<world.GetPrimitiveCount()<<std::endl;
				throw std::runtime_error("N in LAMMPS does not match N in SAPHRON!");
			}

			double *x = new double[3*natoms];
			double *v = new double[3*natoms];
			int *image = new int[natoms];
			lammps_gather_atoms(_lmp, "image", 0, 1, image);
			auto& box = world.GetHMatrix();

			lammps_gather_atoms(_lmp, "x", 1, 3, x);
			lammps_gather_atoms(_lmp, "v", 1, 3, v);
			Position pos;
			for (int i = 0; i < natoms; i++)
			{
				pos[0] = x[i*3] + box(0,0)*((image[i] & IMGMASK) - IMGMAX);
				pos[1] = x[i*3 + 1] + box(1,1)*((image[i] >> IMGBITS & IMGMASK) - IMGMAX);
				pos[2] = x[i*3 + 2] + box(2,2)*((image[i] >> IMG2BITS) - IMGMAX);

				_L2S_map[i]->SetPosition(pos);
				_L2S_vxmap[_L2S_map[i]] = v[i*3]; ///***** could cause problem
				_L2S_vymap[_L2S_map[i]] = v[i*3 + 1]; ///*****
				_L2S_vzmap[_L2S_map[i]] = v[i*3 + 2]; ///*****
			}

			delete [] x;
			delete [] v;
			delete [] image;
		}

	public:
		MDMove(std::string data_file, std::string input_file, 
				std::string minimize_file, std::vector<std::string> sids,
				std::vector<int> lids, unsigned seed = 2437) : 
		_rand(seed), _comm_lammps(), _lmp(), _data_file(data_file), 
		_input_file(input_file), _minimize_file(minimize_file),
		_S2L_map(), _S2L_imap(), _L2S_map()
		{
			MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &_comm_lammps);
			if(sids.size() != lids.size())
			{
				std::cout<< "mapping SAPHRON ids size not same as LAMMPS size, this error shouldn't happen!"<<std::endl;
				exit(1);
			}

			for(int i = 0; i < sids.size(); i++)
				_S2L_imap[sids[i]] = lids[i];

		}

		virtual void Perform(WorldManager* wm, 
					 ForceFieldManager* ffm, 
					 const MoveOverride& override) override
		{
			World* w = wm->GetRandomWorld();
			
			// matches id in spahron to lammps
			UpdateMap(*w);

			// this writes lammps data file
			WriteDataFile(*w);
			
			// Silence of the lammps.
			char **largs = (char**) malloc(sizeof(char*) * 3);
			for(int i = 0; i < 3; ++i)
				largs[i] = (char*) malloc(sizeof(char) * 1024);
			sprintf(largs[0], " ");
			sprintf(largs[1], "-screen");
			sprintf(largs[2], "none");

			_lmp = new LAMMPS(3, largs, _comm_lammps);
			// if minimize file exits lammps will run it
			if(_minimize_file.compare("none") != 0)
			{
				ReadInputFile(_minimize_file);
				_minimize_file = "none";
			}
			else
			{
				ReadInputFile(_input_file);
			}
			// updates positoin in saphron
			UpdateSAPHRON(*w);

			delete _lmp;
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
		void SetOrderParameterPrefactor(bool flag) { }

		virtual double GetAcceptanceRatio() const override
		{
			return 1.0;
		};

		virtual void ResetAcceptanceRatio() override
		{

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
