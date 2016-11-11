#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "stdio.h"
#include <math.h> 
#include "stdlib.h"
#include <string.h>
#include <string>
#include "mpi.h"
#include <omp.h>
#include <time.h>
#include "../src/ForceFields/ForceFieldManager.h"
#include "../src/ForceFields/LennardJonesTSFF.h"
#include "../src/ForceFields/DSFFF.h"
#include "../src/ForceFields/DebyeHuckelFF.h"
#include "../src/ForceFields/FENEFF.h"
#include "../src/ForceFields/HarmonicFF.h"
#include "../src/Moves/MoveManager.h"
#include "../src/Moves/Move.h"
#include "../src/Moves/InsertParticleMove.h"
#include "../src/Moves/DeleteParticleMove.h"
#include "../src/Moves/AcidReactionMove.h"
#include "../src/Moves/AcidTitrationMove.h"
#include "../src/Moves/AnnealChargeMove.h"
#include "../src/Particles/Particle.h"
#include "../src/Worlds/World.h"
#include "../src/Worlds/WorldManager.h"
#include "../src/Properties/Energy.h"

#include "lammps.h"
#include "input.h"
#include "atom.h"
#include "library.h"

using namespace std;
using namespace SAPHRON;
using namespace LAMMPS_NS;

// forward declaration
LAMMPS* Equlibration(std::string lammpsfile, MPI_Comm& lammps_comm);
void setSaphronBondedNeighbors(ParticleList &Monomers);
void ReadInputFile(std::string lammpsfile, MPI_Comm& comm_lammps, LAMMPS* lmp, int& xrand);
void SAPHRONLoop(MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, World &world);//dfinsdkjcnkdsjkjkkjkdfjk
void WriteDataFile(LAMMPS* lmp, World &world, std::ofstream& data_file); // soicndcndjfkjsd
void WriteResults(LAMMPS* lmp, ParticleList &MonomersA, ParticleList &MonomersB, std::ofstream& results_file, double &debye);
void WriteDump(LAMMPS* lmp, World &world, std::ofstream& dump_file, double &debye, int &loop); //dsjkbdjcjhkjbdjhdhjb

// Main code
int main(int narg, char **arg)
{
  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)

  MPI_Init(&narg,&arg);

  int me,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


  int nprocs_lammps = atoi(arg[1]);
  std::string lammpsfile = std::string(arg[2]);
  int numLoops = std::stoi(arg[3]);
  double kappa = atof(arg[4]);
  double coulcut = (1.0/kappa)*5.0;
  double mu = atof(arg[5]);
  double debye = atof(arg[6]);

  std::ofstream dump_file;
  dump_file.open("dump_debyeLen_" + std::to_string(debye)+"_.dat", std::ofstream::out);
  dump_file<<"id   type   q   x   y   z   ix   iy   iz"<<std::endl;
  dump_file.close();
  std::ofstream results_file;
  results_file.open("debyeLen_" + std::to_string(debye)+"_results.dat", std::ofstream::out);
  results_file<<"f_polyA   f_polyB   Rg_polyA   Rg_polyB   Rg_polyAB   Potential_Energy   Total Energy"<<std::endl;
  results_file.close();
  std::ifstream datatrial("data.trial", std::ios::binary);
  std::ofstream dtfile("data."+lammpsfile, std::ios::binary);

  dtfile << datatrial.rdbuf();

  int seed = time(NULL); //seed used for this simulation

  if (nprocs_lammps > nprocs || me >= nprocs_lammps) {
    if (me == 0)
      printf("ERROR: LAMMPS cannot use more procs than available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  MPI_Comm comm_lammps;
  MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &comm_lammps);
  
  srand(time(NULL));
  int xrand = rand()%99999999+1;   
  LAMMPS *lmp;
  lmp = new LAMMPS(0, NULL, comm_lammps);
  ReadInputFile(lammpsfile, comm_lammps, lmp, xrand);

  cout << "I am here"<<endl;
  int natoms = static_cast<int> (lmp->atom->natoms);
  cout << natoms << "this is the number of atoms"<<endl;
  double *x = new double[3*natoms];
  int *type = new int[natoms];
  lammps_gather_atoms(lmp, "x", 1, 3, x);
  lammps_gather_atoms(lmp,"type", 0, 1, type);
  cout << "we are here AI"<<endl;
  
  /////////////////SETUP SAPHRON//////////////////////////////////////////////////////////////////////
  World world(1000.0, 1000.0, 1000.0, coulcut, seed + 1);
  world.SetTemperature(1.0);

  ParticleList MonomersA;
  ParticleList MonomersB;
  SAPHRON::Particle polyA("PolymerA");
  SAPHRON::Particle polyB("PolymerB");

  // CREATE THE TWO POLYMERS
  for(int i=0; i<natoms*3; i=i+3)
  {
    if(type[i/3] == 1)
    {
      MonomersA.push_back(new Particle({x[i],x[i+1],x[i+2]},{0.0,0.0,0.0}, "MonoA"));
    }
    if(type[i/3] == 2)
    {
      MonomersB.push_back(new Particle({x[i],x[i+1],x[i+2]},{0.0,0.0,0.0}, "MonoB"));
    }
  }

  // SET THE CHARGES ON THE MONOMERS TO 0.0
  for(auto& m : MonomersA)
    m->SetCharge(0.0);
  for(auto& m : MonomersB)
    m->SetCharge(0.0);

// SETTING THE NEIGHBORS IN THE SAPHRON WORLD, SINCE LINEAR POLYMERS



int MonomersA_size = (int)double(MonomersA.size());
int MonomersB_size = (int)double(MonomersB.size());

  for(int i=1; i < MonomersA_size-1; i++)
  {
    MonomersA[i]->AddBondedNeighbor(MonomersA[i+1]);
    MonomersA[i]->AddBondedNeighbor(MonomersA[i-1]);
  }
  MonomersA[0]->AddBondedNeighbor(MonomersA[1]);
  MonomersA[MonomersA_size-1]->AddBondedNeighbor(MonomersA[MonomersA_size-2]);


  for(int i=2; i < MonomersB_size-1; i++)
  {
    MonomersB[i]->AddBondedNeighbor(MonomersB[i+1]);
    MonomersB[i]->AddBondedNeighbor(MonomersB[i-1]);
  }
  MonomersB[1]->AddBondedNeighbor(MonomersB[2]);
  MonomersB[0]->AddBondedNeighbor(MonomersB[MonomersB_size-1]);
  MonomersB[MonomersB_size-1]->AddBondedNeighbor(MonomersB[MonomersB_size-2]);
  MonomersB[MonomersB_size-1]->AddBondedNeighbor(MonomersB[0]);

  for(auto& c : MonomersB)
  {
    cout << "for MonomersB species ID" << c->GetGlobalIdentifier() << " bonded to: ";
    for(auto& b : c->GetBondedNeighbors())
      cout<<b->GetGlobalIdentifier()<<" ";
      cout<<endl;  
  }

// ADD MONOMERS AS A CHILD FOR THE POLYMERS 
  for(auto& c : MonomersA)
    polyA.AddChild(c);
  for(auto& c : MonomersB)
    polyB.AddChild(c);


// CREATE FORCEFIELDS
  ForceFieldManager ffm;
  LennardJonesTSFF lj(1.0, 1.0, {2.5}); // Epsilon, sigma, cutoff
  FENEFF fene(0, 1.0, 7.0, 2.0); // epsilon, sigma, kspring, rmax
  DebyeHuckelFF debHuc(kappa, {coulcut});
  ffm.SetElectrostaticForcefield(debHuc);

  ffm.AddNonBondedForceField("MonoA", "MonoA", lj);
  ffm.AddNonBondedForceField("MonoB", "MonoB", lj);
  ffm.AddNonBondedForceField("MonoA", "MonoB", lj);
  ffm.AddBondedForceField("MonoA", "MonoA", lj);
  ffm.AddBondedForceField("MonoA", "MonoA", fene);
  ffm.AddBondedForceField("MonoA", "MonoA", debHuc);
  ffm.AddBondedForceField("MonoB", "MonoB", lj);
  ffm.AddBondedForceField("MonoB", "MonoB", fene);
  ffm.AddBondedForceField("MonoB", "MonoB", debHuc);

  
  // CREATE WORLD
  WorldManager WM;

  WM.AddWorld(&world);
  world.AddParticle(&polyA);
  world.AddParticle(&polyB);

  // SET UP MOVES
  MoveManager MM (seed);
  AnnealChargeMove AnnMvA({{"PolymerA"}}, seed + 2);
  AnnealChargeMove AnnMvB({{"PolymerB"}}, seed + 2);
  AcidTitrationMove AcidTitMvA({{"MonoA"}}, 1.67, mu, seed + 6);  //bjerrum length 2.8sigma (e*sqrt(2.8) == 1.67)
  AcidTitrationMove AcidTitMvB({{"MonoB"}}, -1.67, mu, seed + 6); // REPRESENTING POLYBASE

  MM.AddMove(&AnnMvA);
  MM.AddMove(&AnnMvB);
  MM.AddMove(&AcidTitMvA);
  MM.AddMove(&AcidTitMvB);

  delete lmp;
  delete [] x;

  //Make sure all threads are caught up to this point.
  MPI_Barrier(MPI_COMM_WORLD);

  // Run SAPHRON and write out results and new data file for new LAMMPS instance
  xrand++;
  int hit_detection_numb = 0;
  for(int loop=0; loop<numLoops; loop++)   
  {
    lmp = new LAMMPS(0, NULL, comm_lammps);
    ReadInputFile(lammpsfile, comm_lammps, lmp, xrand);
    natoms = static_cast<int> (lmp->atom->natoms);
    double *x = new double[3*natoms];
    int *type = new int[natoms];
    lammps_gather_atoms(lmp,"x",1,3,x);
    lammps_gather_atoms(lmp,"type", 0, 1, type);

    // Set position of monomers
    for(int i=0; i<natoms*3; i=i+3)
    {
      if(type[i/3] == 1)
      {
        MonomersA[int(i/3)]->SetPosition({x[i],x[i+1],x[i+2]});
      }
    }
    // check here if this is right
    int kk = 0;
    for(int i=0; i<natoms*3; i=i+3)
    {
      if(type[i/3] == 2)
      {
        MonomersB[kk]->SetPosition({x[i],x[i+1],x[i+2]});
        kk++;
      }
    }

    SAPHRONLoop(MM, WM, ffm, world);

    int rank;
    MPI_Comm_rank(comm_lammps,&rank);
    if(rank == 0)
    {
      std::ofstream data_file;
      data_file.open("data."+lammpsfile, std::ofstream::out);
      WriteDataFile(lmp, world, data_file);
      data_file.close();
      WriteResults(lmp, MonomersA, MonomersB, results_file, debye);
      if (hit_detection_numb == 9)
      {
      	WriteDump(lmp, world, dump_file, debye, loop);
      	hit_detection_numb = -1;
      }
      
    }
    xrand++;
    hit_detection_numb++;
    delete lmp;
    delete [] x;
  }

  // close down MPI
  MPI_Finalize();
}

void ReadInputFile(std::string lammpsfile, MPI_Comm& comm_lammps, LAMMPS* lmp, int& xrand)
{
  // open LAMMPS input script
  FILE *fp;
  int rank;
  MPI_Comm_rank(comm_lammps,&rank);
  if (rank == 0)
  {
    fp = fopen(lammpsfile.c_str(),"r");
    if (fp == NULL)
    {
      printf("ERROR: Could not open LAMMPS input script\n");
      MPI_Abort(MPI_COMM_WORLD,1);
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

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if (n == 0) break;
    MPI_Bcast(line,n,MPI_CHAR,0,MPI_COMM_WORLD);

    char *found;
    found = strstr(line, "langevin");

    if (found!=NULL)
    {
      string random = "fix  2 all langevin 1.0 1.0 100.0 "+std::to_string(xrand);
      strcpy(line,random.c_str());
    }
    lmp->input->one(line);
  }
}

void SAPHRONLoop(MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, World &world)
{
  // Energy evaluation and setting
  auto H1 = ffm.EvaluateEnergy(world);
  world.SetEnergy(H1.energy);
  world.UpdateNeighborList();

  // Perform moves for M steps ()
  for(int i=0; i<10; i++)
  {
    auto* move = MM.SelectRandomMove();
    move->Perform(&WM, &ffm, MoveOverride::None);
  }    
}

// WRITE THE RESULTS
void WriteResults(LAMMPS* lmp, ParticleList &MonomersA, ParticleList &MonomersB, std::ofstream& results_file, double &debye)
{
  results_file.open("debyeLen_" + std::to_string(debye)+"_results.dat", std::ofstream::app);
  int sumChargeA = 0;
  int sumChargeB = 0;
  for(auto& p : MonomersA)
  {
    if (p->GetCharge() < 0)
      sumChargeA++;
  }
  for(auto& p : MonomersB)
  {
    if (p->GetCharge() < 0)
      sumChargeB++;
  }
  double Rg_polyA_value = *((double*) lammps_extract_compute(lmp, "Rg_compute_A", 0, 0));
  double Rg_polyB_value = *((double*) lammps_extract_compute(lmp, "Rg_compute_B", 0, 0));
  double Rg_polyAB_value = *((double*) lammps_extract_compute(lmp, "Rg_compute_All", 0, 0));
  double PE_value = *((double*) lammps_extract_compute(lmp, "myPE", 0, 0));
  double KE_value = *((double*) lammps_extract_compute(lmp, "myKE", 0, 0));
  double Total_E_value = PE_value + KE_value;
  double f_valueA = double(sumChargeA)/double(MonomersA.size());
  double f_valueB = double(sumChargeB)/double(MonomersB.size());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
  {
  results_file<<f_valueA<<" "<<f_valueB<<" "<<Rg_polyA_value<<" "<<Rg_polyB_value<<" "<<Rg_polyAB_value<<" "<<PE_value<<" "<<Total_E_value<<std::endl;
  results_file.close();
  }
}

//  WRITE THE LAMMPS DATA FILE
void WriteDataFile(LAMMPS* lmp, World &world, std::ofstream& ofs)
{
  int natoms = static_cast<int> (lmp->atom->natoms);
  double *vel = new double[3*natoms]; 
  int *image = new int[natoms];
  int *image_all = new int[3*natoms];
  lammps_gather_atoms(lmp, "image", 0, 1, image);
  lammps_gather_atoms(lmp, "v", 1, 3, vel);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
  {
    for (int i = 0; i < natoms; i++)
    {
      image_all[i*3] = (image[i] & IMGMASK) - IMGMAX;;
      image_all[i*3+1] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      image_all[i*3+2] = (image[i] >> IMG2BITS) - IMGMAX;
    }

    int numlammpsatoms;
    std::string garbage;

    //Read in file and change what is needed
    std::ifstream infile("data.trial");
    std::string line;
    while (std::getline(infile, line))
    {
      if(line.empty())
        continue;

      std::istringstream iss(line);
      std::string s2 = iss.str();
      
      std::string s1 = "atoms";
      if (s2.std::string::find(s1) != std::string::npos)
      {
        ofs<<"       "<<std::to_string(natoms)<<" atoms"<<std::endl;
        iss >> numlammpsatoms >> garbage;
        std::cout<<"Num lammps atoms is "<<numlammpsatoms<<std::endl;
        continue;
      }
      std::string s3 = "Atoms";
      if (s2.std::string::find(s3) != std::string::npos)
      {
        ofs<<"Atoms"<<std::endl;
        ofs<<std::endl;

        int ii = 0;
        while(ii < numlammpsatoms) // this while loop is important
          // this makes sure that you don't read lines corresponding to previous lammps coordinates
        {
          std::getline(infile,line);
          if(line.empty())
            continue;
          ii++;
        }

          int i = 0;
          for(const auto& p : world)
          {
            if(p->HasChildren())
            {
              for(auto& pp : p->GetChildren())
              {
                ofs<<i+1<<" 1 "<<pp->GetSpeciesID()<<" "<<pp->GetCharge()<<" ";
                for(auto& x : pp->GetPosition()){
                    ofs<<x<<" ";
                }
                ofs<<image_all[i*3]<<" "<<image_all[i*3+1]<<" "<<image_all[i*3+2];
                ofs<<std::endl;
                i++;
              }
            }
          }
        continue;
      }

      std::string s4 = "Masses";
      if (s2.std::string::find(s4) != std::string::npos)
      {
        ofs<<"Masses"<<std::endl;
        ofs<<std::endl;
        continue;
      }

      /*std::string vstr = "Velocities";
      if (s2.std::string::find(vstr) != std::string::npos)
      {
        ofs<<"Velocities"<<std::endl;
        ofs<<std::endl;

        int ii = 0;
        while(ii < numlammpsatoms) // this while loop is important
          // this makes sure that you don't read lines corresponding to previous lammps coordinates
        {
          std::getline(infile,line);
          if(line.empty())
            continue;
          ii++;
        }

        for(int i = 0; i < atoms.size(); i++)
        {
          ofs<<i+1<<" "<<vel[i*3]<<" "<<vel[i*3+1]<<" "<<vel[i*3+2];
          ofs<<std::endl;
        }
        continue;
      }*/
      
      std::string s5 = "Bonds";
      if (s2.std::string::find(s5) != std::string::npos)
      {
        ofs<<"Bonds"<<std::endl;
        ofs<<std::endl;
        continue;
      }

      std::string s6 = "xlo";
      if (s2.std::string::find(s6) != std::string::npos)
      {
        ofs<<"       0.0 1000.0 xlo xhi"<<std::endl;
        continue;
      }

      std::string s7 = "ylo";
      if (s2.std::string::find(s7) != std::string::npos)
      {
        ofs<<"       0.0 1000.0 ylo yhi"<<std::endl;
        continue;
      }

      std::string s8 = "zlo";
      if (s2.std::string::find(s8) != std::string::npos)
      {
        ofs<<"       0.0 1000.0 zlo zhi"<<std::endl;
        continue;
      }

      ofs<<iss.str()<<std::endl;
    }
    ofs.close();
}
  delete [] image;
  delete [] image_all;
  delete [] vel;
}


void WriteDump(LAMMPS* lmp, World &world, std::ofstream& dump_file, double &debye, int &loop)
{
  int natoms = static_cast<int> (lmp->atom->natoms);
  int *image = new int[natoms];
  int *image_all = new int[3*natoms];
  lammps_gather_atoms(lmp, "image", 0, 1, image);

  for (int i = 0; i < natoms; i++)
  {
    image_all[i*3] = (image[i] & IMGMASK) - IMGMAX;;
    image_all[i*3+1] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
    image_all[i*3+2] = (image[i] >> IMG2BITS) - IMGMAX;
  }

  // WRITE PRELIMINARY THINGS
  int MDstep = loop*1000;
  dump_file.open("dump_debyeLen_" + std::to_string(debye)+"_.dat", std::ofstream::app);
  dump_file<<"################!!!!!!The MD step is: "<<MDstep<<" ################!!!!!!!!!!"<<std::endl;

  int i = 0;
  for(const auto& p : world)
  {
    if(p->HasChildren())
    {
      for(auto& pp : p->GetChildren())
      {
        dump_file<<i+1<<" "<<pp->GetSpeciesID()<<" "<<pp->GetCharge()<<" ";
        for(auto& x : pp->GetPosition()){
            dump_file<<x<<" ";
        }
        dump_file<<image_all[i*3]<<" "<<image_all[i*3+1]<<" "<<image_all[i*3+2];
        dump_file<<std::endl;
        i++;
      }
    }
  }
  dump_file.close();

  delete [] image;
  delete [] image_all; 
}