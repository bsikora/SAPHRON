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
#include <iomanip>
#include <stdio.h>

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

		std::map<int, int> _S2L_ixmap; ///*****
		std::map<int, int> _S2L_iymap; ///*****
		std::map<int, int> _S2L_izmap; ///*****

		std::map<std::string, int> _S2L_imap;
		std::map<int, Particle*> _L2S_map;

		std::map<Particle*, double> _L2S_vxmap; ///*****
		std::map<Particle*, double> _L2S_vymap; ///*****
		std::map<Particle*, double> _L2S_vzmap; ///*****

		std::map<Particle*, int> _L2S_ixmap; ///*****
		std::map<Particle*, int> _L2S_iymap; ///*****
		std::map<Particle*, int> _L2S_izmap; ///*****		

		int _bondnumber;
		int _atomnumber;
		double _wallspace_y = -0.000000; ///***** 0.000001
		double _wallspace_z = -0.000000; ///***** 0.000001
		int _perma_mark = 0; ///*****

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

						_S2L_ixmap[cp->GetGlobalIdentifier()] = _L2S_ixmap[cp]; ///*****
						_S2L_iymap[cp->GetGlobalIdentifier()] = _L2S_iymap[cp]; ///*****
						_S2L_izmap[cp->GetGlobalIdentifier()] = _L2S_izmap[cp]; ///*****

						lammps_id++;
					}
				}
				else
				{
					_S2L_map[p->GetGlobalIdentifier()] = lammps_id;
					_S2L_vxmap[p->GetGlobalIdentifier()] = _L2S_vxmap[p]; ///*****
					_S2L_vymap[p->GetGlobalIdentifier()] = _L2S_vymap[p]; ///*****
					_S2L_vzmap[p->GetGlobalIdentifier()] = _L2S_vzmap[p]; ///*****

					_S2L_ixmap[p->GetGlobalIdentifier()] = _L2S_ixmap[p]; ///*****
					_S2L_iymap[p->GetGlobalIdentifier()] = _L2S_iymap[p]; ///*****
					_S2L_izmap[p->GetGlobalIdentifier()] = _L2S_izmap[p]; ///*****

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

			auto ix = _S2L_ixmap[p->GetGlobalIdentifier()]; ///*****
			auto iy = _S2L_iymap[p->GetGlobalIdentifier()]; ///*****
			auto iz = _S2L_izmap[p->GetGlobalIdentifier()]; ///*****

			Position ppos = p->GetPosition();

			/*auto& H = p->GetWorld()->GetHMatrix(); 
			if(ppos[0] >= H(0,0) || ppos[1] >= H(1,1) || ppos[2] >= H(2,2))
				std::cerr << "Error with particle " << p->GetGlobalIdentifier() << " " << ppos << std::endl;*/

			coords += std::to_string(pid) + " 1 " + 
					std::to_string(sid) + " " +
					std::to_string(p->GetCharge()) + " " +
					std::to_string(ppos[0]) + " " +
					std::to_string(ppos[1]) + " " +
					std::to_string(ppos[2]) + " " + 
					std::to_string(ix) + " " + std::to_string(iy) + " " + std::to_string(iz) + "\n";

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
			/*datafile<<"0 "+std::to_string(box(0,0)) +" xlo xhi\n";
			datafile<<"0 "+std::to_string(box(1,1)) +" ylo yhi\n";  ///*****previous 700
			datafile<<"0 "+std::to_string(box(2,2)) +" zlo zhi\n\n";*/

			datafile<<"0 "+std::to_string(box(0,0)) +" xlo xhi\n";
			datafile<<std::to_string(-_wallspace_y)+" "+std::to_string(box(1,1)+_wallspace_y) +" ylo yhi\n";  ///*****
			datafile<<std::to_string(-_wallspace_z)+" "+std::to_string(box(2,2)+_wallspace_z) +" zlo zhi\n\n"; ///*****

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
					throw std::logic_error("ERROR: Could not open LAMMPS input scripts " + file_to_read);
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
					test.replace(f, std::string("RANDOM").length(), std::to_string(int(_rand.int32()/1000 + 1)));///*****
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
				/*
				pos[0] = x[i*3] + box(0,0)*((image[i] & IMGMASK) - IMGMAX);
				pos[1] = x[i*3 + 1] + (box(1,1)+2*_wallspace_y)*((image[i] >> IMGBITS & IMGMASK) - IMGMAX); ///*****
				pos[2] = x[i*3 + 2] + (box(2,2)+2*_wallspace_z)*((image[i] >> IMG2BITS) - IMGMAX); ///*****
				*/
				pos[0] = x[i*3];
				pos[1] = x[i*3 + 1]; ///*****
				pos[2] = x[i*3 + 2]; ///*****

				/*
				std::cerr << "position from lammps x " << x[i*3] << " " << std::endl;      ///*****
				std::cerr << "position from lammps y " << x[i*3 + 1] << " " << std::endl;  ///*****
				std::cerr << "position from lammps z " << x[i*3 + 2] << " " << std::endl; ///*****
				*/

				_L2S_map[i]->SetPosition(pos);
				_L2S_vxmap[_L2S_map[i]] = v[i*3]; ///***** could cause problem
				_L2S_vymap[_L2S_map[i]] = v[i*3 + 1]; ///*****
				_L2S_vzmap[_L2S_map[i]] = v[i*3 + 2]; ///*****

				_L2S_ixmap[_L2S_map[i]] = ((image[i] & IMGMASK) - IMGMAX); ///***** could cause problem
				_L2S_iymap[_L2S_map[i]] = ((image[i] >> IMGBITS & IMGMASK) - IMGMAX); ///*****
				_L2S_izmap[_L2S_map[i]] = ((image[i] >> IMG2BITS) - IMGMAX); ///*****
			}

	///*****// NOTE THIS METHOD IS BUILT FOR SINGLE POLYMER IN THE SYSTEM ONLY, FOR DOUBLE POLYMER FURTHER CHANGES WOULD BE REQUIRED
			// PRINTING OUT RG AND CHARGE FRAC VALUES
			if (_perma_mark == 0)
			{
				for (auto& p : world)
				{
					if(p->HasChildren())
					{
						for(auto& cp : *p)
						{
							//std::cout<< "species ID is"<< cp->GetSpecies()<<std::endl;
						    if ((cp->GetSpecies().compare("Monomer2") == 0) || (cp->GetSpecies().compare("dMonomer2") == 0))
						    {
	      						_perma_mark = 2;
	      						break;
	  						}else{
	  							_perma_mark = 1;
	  						}
	      				}
	  				}
				}
			}
			//std::cout<< "perma_mark is"<< _perma_mark<<std::endl;

			if (_perma_mark == 1)
			{
				/* code */
			
			int sumCharge = 0;
			int num_monomers = 0;
			for(auto& p : world)
				if(p->HasChildren())
				{
					num_monomers = 0;
					sumCharge = 0;
					for(auto& cp : *p)
					{
						num_monomers++;
					    if ((cp->GetCharge() < 0) || (cp->GetCharge() > 0))
      						sumCharge++;
      				}
  				}

		  	double f_value = double(sumCharge)/double(num_monomers);
			double Rg_value = *((double*) lammps_extract_compute(_lmp, "Rg_compute", 0, 0));
			double PE_value = *((double*) lammps_extract_compute(_lmp, "myPE", 0, 0));
			double KE_value = *((double*) lammps_extract_compute(_lmp, "myKE", 0, 0));
			double Total_E_value = PE_value + KE_value;

			std::ofstream Rgchgfracfile;
			Rgchgfracfile.open("Rg_chg_frac_"+_input_file,std::ofstream::app);
		  	Rgchgfracfile<<std::setprecision(4)<<std::fixed<<std::to_string(f_value)<<"     "<<std::to_string(Rg_value)<<
		  	"     "<<std::to_string(PE_value)<<"     "<<std::to_string(Total_E_value)<<"     "<<std::to_string(num_monomers)<<
		  	"     "<<std::to_string(sumCharge)<<std::endl;
		  	Rgchgfracfile.close();
		  }

			if (_perma_mark == 2)
			{
				/* code */
			
			int sumCharge_1 = 0;
			int sumCharge_2 = 0;
			int num_monomers_1 = 0;
			int num_monomers_2 = 0;
			int _count = 0;
			for(auto& p : world)
			{
				if(p->HasChildren())
				{
					for(auto& cp : *p)
					{
						if ((cp->GetSpecies().compare("Monomer2") == 0) || (cp->GetSpecies().compare("dMonomer2") == 0))
							num_monomers_2++;
						if ((cp->GetSpecies().compare("Monomer") == 0) || (cp->GetSpecies().compare("dMonomer") == 0))
							num_monomers_1++;
					    if ((cp->GetSpecies().compare("dMonomer2") == 0) || (cp->GetCharge() > 0))
      						sumCharge_2++;
					    if ((cp->GetSpecies().compare("dMonomer") == 0) || (cp->GetCharge() < 0))
      						sumCharge_1++;
      				}
  				}
  				_count++;
  			}

		  	double f_value_mono_1 = double(sumCharge_1)/double(num_monomers_1);
		  	double f_value_mono_2 = double(sumCharge_2)/double(num_monomers_2);
			double Rg_value_mono_1 = *((double*) lammps_extract_compute(_lmp, "Rg_compute_1", 0, 0));
			double Rg_value_mono_2 = *((double*) lammps_extract_compute(_lmp, "Rg_compute_2", 0, 0));
			double PE_value = *((double*) lammps_extract_compute(_lmp, "myPE", 0, 0));
			double KE_value = *((double*) lammps_extract_compute(_lmp, "myKE", 0, 0));
			double Total_E_value = PE_value + KE_value;

			std::ofstream Rgchgfracfile;
			Rgchgfracfile.open("Rg_chg_frac_"+_input_file,std::ofstream::app);
		  	Rgchgfracfile<<std::setprecision(4)<<std::fixed<<
		  	std::to_string(f_value_mono_1)<<
		  	"     "<<std::to_string(Rg_value_mono_1)<<
		  	"     "<<std::to_string(f_value_mono_2)<<
		  	"     "<<std::to_string(Rg_value_mono_2)<<
		  	"     "<<std::to_string(PE_value)<<
		  	"     "<<std::to_string(Total_E_value)<<
		  	"     "<<std::to_string(num_monomers_1)<<
		  	"     "<<std::to_string(num_monomers_2)<<
		  	"     "<<std::to_string(sumCharge_1)<<
		  	"     "<<std::to_string(sumCharge_2)<<std::endl;
		  	Rgchgfracfile.close();
		  }
  	///*****

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

			// REMOVING THE RG_CHARGE_FRACTION FILE AT THE BEGINNING OF THE RUN
			remove(("Rg_chg_frac_"+_input_file).c_str()); ///*****

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
			sprintf(largs[2], "none");  ///*****

             ///replace 0 with 3 and NULL with largs to suppress lammps output
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
