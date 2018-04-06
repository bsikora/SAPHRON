#pragma once 

#include "Move.h"
#include <map>
#include "../Utils/Helpers.h"
#include "../Utils/Rand.h"
#include "../Worlds/WorldManager.h"
#include "../ForceFields/ForceFieldManager.h"
#include "../DensityOfStates/DOSOrderParameter.h"

// This move currently only supports **one** type of bond
namespace SAPHRON
{
	class MDMoveMultiCore : public Move
	{
	private:

		Rand _rand;
		std::string _data_file;
		std::string _input_file;
		std::string _minimize_file;

		std::string _ssages_file;
		bool _ssages_check = false;
		int _num_proc = 1; // initialized through move.cpp

		std::map<int, int> _S2L_map;

		std::map<int, double> _S2L_vxmap;
		std::map<int, double> _S2L_vymap;
		std::map<int, double> _S2L_vzmap;

		std::map<int, int> _S2L_ixmap;
		std::map<int, int> _S2L_iymap;
		std::map<int, int> _S2L_izmap;

		std::map<std::string, int> _S2L_imap;
		std::map<int, Particle*> _L2S_map;

		std::map<Particle*, double> _L2S_vxmap;
		std::map<Particle*, double> _L2S_vymap;
		std::map<Particle*, double> _L2S_vzmap;

		std::map<Particle*, int> _L2S_ixmap;
		std::map<Particle*, int> _L2S_iymap;
		std::map<Particle*, int> _L2S_izmap;		

		int _bondnumber;
		int _atomnumber;
		double _wallspace_y = 0.0000001;
		double _wallspace_z = 0.0000001;
		int _perma_mark = 0;

		int MDsweeps = 0;
		int hit_detection_numb = 0;
		int Rg_chg_hit_detection_numb = 0;

		std::string _exe;
		std::string _in_str;




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
						_S2L_vxmap[cp->GetGlobalIdentifier()] = _L2S_vxmap[cp];
						_S2L_vymap[cp->GetGlobalIdentifier()] = _L2S_vymap[cp];
						_S2L_vzmap[cp->GetGlobalIdentifier()] = _L2S_vzmap[cp];

						_S2L_ixmap[cp->GetGlobalIdentifier()] = _L2S_ixmap[cp];
						_S2L_iymap[cp->GetGlobalIdentifier()] = _L2S_iymap[cp];
						_S2L_izmap[cp->GetGlobalIdentifier()] = _L2S_izmap[cp];

						lammps_id++;
					}
				}
				else
				{
					_S2L_map[p->GetGlobalIdentifier()] = lammps_id;
					_S2L_vxmap[p->GetGlobalIdentifier()] = _L2S_vxmap[p];
					_S2L_vymap[p->GetGlobalIdentifier()] = _L2S_vymap[p];
					_S2L_vzmap[p->GetGlobalIdentifier()] = _L2S_vzmap[p];

					_S2L_ixmap[p->GetGlobalIdentifier()] = _L2S_ixmap[p];
					_S2L_iymap[p->GetGlobalIdentifier()] = _L2S_iymap[p];
					_S2L_izmap[p->GetGlobalIdentifier()] = _L2S_izmap[p];

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

			auto vx = _S2L_vxmap[p->GetGlobalIdentifier()];
			auto vy = _S2L_vymap[p->GetGlobalIdentifier()];
			auto vz = _S2L_vzmap[p->GetGlobalIdentifier()];

			auto ix = _S2L_ixmap[p->GetGlobalIdentifier()];
			auto iy = _S2L_iymap[p->GetGlobalIdentifier()];
			auto iz = _S2L_izmap[p->GetGlobalIdentifier()];

			Position ppos = p->GetPosition();

			coords += std::to_string(pid) + " 1 " + 
					std::to_string(sid) + " " +
					std::to_string(p->GetCharge()) + " " +
					std::to_string(ppos[0]) + " " +
					std::to_string(ppos[1]) + " " +
					std::to_string(ppos[2]) + " " + 
					std::to_string(ix) + " " + std::to_string(iy) + " " + std::to_string(iz) + "\n";

			velocities += std::to_string(pid) + " " +
					std::to_string(vx) + " " +
					std::to_string(vy) + " " +
					std::to_string(vz) + "\n";

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
			std::string velocities = "Velocities\n\n";
			
			_bondnumber = 0;
			_atomnumber = 0;

			for(auto& p : world)
				if(p->HasChildren())
					for(auto& cp : *p)
						AnalyzeParticle(cp, coords, bonding, velocities);
				else
					AnalyzeParticle(p, coords, bonding, velocities);

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
			datafile<<std::to_string(-_wallspace_y)+" "+std::to_string(box(1,1)+_wallspace_y) +" ylo yhi\n";
			datafile<<std::to_string(-_wallspace_z)+" "+std::to_string(box(2,2)+_wallspace_z) +" zlo zhi\n\n";

			datafile<<"Masses\n\n";
			for(int i = 0; i < world.GetComposition().size(); i++)
				datafile<<std::to_string(i+1) + " 1.0\n";
			datafile<<std::endl;
			datafile<<coords;
			datafile<<velocities;
			datafile<<bonding;

			datafile.close();
		}




		// Update SAPHRON from LAMMPS
		void ReadInputFile(std::string file_to_read, int _num_proc)
		{
			
			FILE *fp;
			fp = fopen(file_to_read.c_str(),"r");
			if (fp == NULL)
			{
				throw std::logic_error("ERROR: Could not open LAMMPS input scripts " + file_to_read);
			}

			std::ofstream inputfile;
			inputfile.open(_input_file+"_seed_update");

			// Read lammps file line by line
			int n;
			char line[1024];
			while (1) 
			{

				if (fgets(line,1024,fp) == NULL) n = 0;
				else n = strlen(line) + 1;
				if (n == 0) fclose(fp);
				if (n == 0) break;

				std::string test = line;

				size_t f = test.find("RANDOM");
				if (f != std::string::npos)
				{
					test.replace(f, std::string("RANDOM").length(), std::to_string(int(_rand.int32()/1000 + 1)));
				}
				inputfile<<test.c_str();
			}

			inputfile.close();

			_exe = "mpirun -np "+std::to_string(_num_proc)+" /opt/crc/l/lammps/11Aug17/src/lammps/src1/lmp_mpi < ";
			_in_str = _exe+_input_file+"_seed_update"+" "+"-screen none";
			std::system(_in_str.c_str());
		}




		// Update SAPHRON from LAMMPS
		void UpdateSAPHRON(const World &world)
		{

			int id, mol, type, ix_read, iy_read, iz_read;
			double charge_read, x_read, y_read, z_read, vx_read, vy_read, vz_read;
			Position pos;

			FILE *fp;
			std::string dumpfile_to_read = "dump_"+_input_file+".lammpstrj";

			fp = fopen(dumpfile_to_read.c_str(),"r");
			if (fp == NULL)
			{
				throw std::logic_error("ERROR: Could not open dump file " + dumpfile_to_read);
			}

			// Read file line by line
			int n, m=0;
			char line[1024];
			while (1) 
			{

				if (fgets(line,1024,fp) == NULL) n = 0;
				else n = strlen(line) + 1;
				if (n == 0) fclose(fp);
				if (n == 0) break;
				
				if (m <= 8)  // this is because to skip first few lines
				{
					m++;
					continue;
				}
				sscanf(line, "%d %d %lf %lf %lf %lf %d %d %d %lf %lf %lf", &id, &type, &charge_read, &x_read, &y_read, &z_read, &ix_read, &iy_read, &iz_read,
					&vx_read, &vy_read, &vz_read);

				pos[0] = x_read;
				pos[1] = y_read;
				pos[2] = z_read;
				_L2S_map[id-1]->SetPosition(pos);
				_L2S_vxmap[_L2S_map[id-1]] = vx_read;
				_L2S_vymap[_L2S_map[id-1]] = vy_read;
				_L2S_vzmap[_L2S_map[id-1]] = vz_read;

				_L2S_ixmap[_L2S_map[id-1]] = ix_read;
				_L2S_iymap[_L2S_map[id-1]] = iy_read;
				_L2S_izmap[_L2S_map[id-1]] = iz_read;

			}
		}



	public:
		MDMoveMultiCore(std::string data_file, std::string input_file, 
				std::string minimize_file,
				int num_proc,
				std::vector<std::string> sids,
				std::vector<int> lids, unsigned seed = 2437) : 
		_rand(seed), _data_file(data_file), 
		_input_file(input_file), _minimize_file(minimize_file),
		_num_proc(num_proc),
		_S2L_map(), _S2L_imap(), _L2S_map()
		{
			if(sids.size() != lids.size())
			{
				std::cout<< "mapping SAPHRON ids size not same as LAMMPS size, this error shouldn't happen!"<<std::endl;
				exit(1);
			}

			for(int i = 0; i < sids.size(); i++)
				_S2L_imap[sids[i]] = lids[i];

			// REMOVING THE RG_CHARGE_FRACTION FILE AND DUMP FILE AT THE BEGINNING OF THE RUN
			remove(("Rg_chg_frac_"+_input_file).c_str());
			remove(("Appended_snapshots_"+_input_file).c_str());

		}

		virtual void Perform(WorldManager* wm, 
					 ForceFieldManager* ffm, 
					 const MoveOverride& override) override
		{
			World* w = wm->GetRandomWorld();

			// matches id in spahron to lammps
			UpdateMap(*w);
			WriteDataFile(*w);
			
			// if minimize file exits lammps will run it
			if(_minimize_file.compare("none") != 0)
			{
				ReadInputFile(_minimize_file, _num_proc);
				_minimize_file = "none";
			}
			else
			{
				ReadInputFile(_input_file, _num_proc);
			}

			// updates positoin in saphron
			UpdateSAPHRON(*w);
			w->UpdateNeighborList();
			// Update energies and pressures.
			auto wor_ef = ffm->EvaluateEnergy(*w);
			w->SetEnergy(wor_ef.energy);
			w->SetPressure(wor_ef.pressure);

		}

		virtual void Perform(World* w, 
							 ForceFieldManager* ffm, 
							 DOSOrderParameter*, 
							 const MoveOverride&) override
		{

			// matches id in spahron to lammps
			UpdateMap(*w);
			WriteDataFile(*w);
			
			// if minimize file exits lammps will run it
			if(_minimize_file.compare("none") != 0)
			{
				ReadInputFile(_minimize_file, _num_proc);
				_minimize_file = "none";
			}
			else
			{
				ReadInputFile(_input_file, _num_proc);
			}
			// updates positoin in saphron
			UpdateSAPHRON(*w);
			w->UpdateNeighborList();

			// Update energies and pressures.
			auto wor_ef = ffm->EvaluateEnergy(*w);
			w->SetEnergy(wor_ef.energy);
			w->SetPressure(wor_ef.pressure);
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
			return new MDMoveMultiCore(static_cast<const MDMoveMultiCore&>(*this));
		}

	};
}