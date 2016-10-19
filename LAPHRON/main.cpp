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
void SAPHRONLoop(MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, World &world);
void WriteDataFile(LAMMPS* lmp, ParticleList &atoms, std::ofstream& data_file);
void WriteResults(LAMMPS* lmp, ParticleList &Monomers, std::ofstream& results_file, double &debye);

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

  std::ofstream results_file;
  results_file.open("debyeLen_" + std::to_string(debye)+"_results.dat", std::ofstream::out);
  results_file<<"f   Rg   Potential_Energy"<<std::endl;
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

  int natoms = static_cast<int> (lmp->atom->natoms);
  double *x = new double[3*natoms];
  double *q = new double[natoms];
  lammps_gather_atoms(lmp, "x", 1, 3, x);
  lammps_gather_atoms(lmp, "q", 1, 1, q);
  
  /////////////////SETUP SAPHRON//////////////////////////////////////////////////////////////////////
  ParticleList Monomers;
  SAPHRON::Particle poly("Polymer");
  // Intialize monomers 
  for(int i=0; i<natoms*3; i=i+3)
    Monomers.push_back(new Particle({x[i],x[i+1],x[i+2]},{0.0,0.0,0.0}, "Monomer"));

  // Set charges on the monomers
  for(int i=0; i<Monomers.size(); i++)
    Monomers[i]->SetCharge(q[i]);

  /*for(int i=1; i < natoms-1; i++)
  {
    Monomers[i]->AddBondedNeighbor(Monomers[i+1]);
    Monomers[i]->AddBondedNeighbor(Monomers[i-1]);
  }
  Monomers[0]->AddBondedNeighbor(Monomers[1]);
  Monomers[natoms-1]->AddBondedNeighbor(Monomers[natoms-2]);*/


  // SET BONDED NEIGHBORS  IN ANOTHER FUNCTION by reading the initial data file
  setSaphronBondedNeighbors(Monomers);

  for(auto& c : Monomers)
    poly.AddChild(c);

  //Create forcefields
  ForceFieldManager ffm;
  LennardJonesTSFF lj(1.0, 1.0, {2.5}); // Epsilon, sigma, cutoff
  FENEFF fene(0, 1.0, 7.0, 2.0); // epsilon, sigma, kspring, rmax
  DebyeHuckelFF debHuc(kappa, {coulcut});
  ffm.SetElectrostaticForcefield(debHuc);
  ffm.AddNonBondedForceField("Monomer", "Monomer", lj);
  ffm.AddBondedForceField("Monomer", "Monomer", lj); // changed bonded force field
  ffm.AddBondedForceField("Monomer", "Monomer", fene); // changed bonded force field
  ffm.AddBondedForceField("Monomer", "Monomer", debHuc); // changed bonded force field

  //Create world
  WorldManager WM;
  double rcut = 300.0;
  World world(1000.0, 1000.0, 1000.0, rcut, seed + 1); // same as lammps input data file
  world.SetTemperature(1.0);
  WM.AddWorld(&world);
  world.AddParticle(&poly);

  //Set up moves
  MoveManager MM (seed);
  AnnealChargeMove AnnMv({{"Polymer"}}, seed + 2);
  //InsertParticleMove Ins({{"Monomer"}}, WM,20,false,seed + 3);
  //DeleteParticleMove Del({{"Monomer"}},false, seed + 4);
  //AcidReactionMove AcidMv({{"Monomer"}}, {{"temp"}},WM,20,10, seed + 5);
  AcidTitrationMove AcidTitMv({{"Monomer"}}, 1.67, mu, seed + 6);    //bjerrum length 2.8sigma (e*sqrt(2.8) == 1.67)

  MM.AddMove(&AnnMv);
  //MM.AddMove(&Ins);
  //MM.AddMove(&Del);
  //MM.AddMove(&AcidMv);
  MM.AddMove(&AcidTitMv);

  delete lmp;
  delete [] x;
  delete [] q;

  //Make sure all threads are caught up to this point.
  MPI_Barrier(MPI_COMM_WORLD);

  // Run SAPHRON and write out results and new data file for new LAMMPS instance
  xrand++;
  for(int loop=0; loop<numLoops; loop++)   
  {
    lmp = new LAMMPS(0, NULL, comm_lammps);
    ReadInputFile(lammpsfile, comm_lammps, lmp, xrand);
    double *x = new double[3*natoms];
    lammps_gather_atoms(lmp,"x",1,3,x);

    // Set position of monomers
    for(int i=0; i<natoms*3; i=i+3)
      Monomers[int(i/3)]->SetPosition({x[i],x[i+1],x[i+2]});

    SAPHRONLoop(MM, WM, ffm, world);

    int rank;
    MPI_Comm_rank(comm_lammps,&rank);
    if(rank == 0)
    {
      std::ofstream data_file;
      data_file.open("data."+lammpsfile, std::ofstream::out);
      WriteDataFile(lmp, Monomers, data_file);
      data_file.close();
      WriteResults(lmp, Monomers, results_file, debye);
    }
    xrand++;
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
      cout <<"MLG PRO 360 YY NO SCOPE !!!!!!! "<<line <<endl;
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

void WriteResults(LAMMPS* lmp, ParticleList &Monomers, std::ofstream& results_file, double &debye)
{
  results_file.open("debyeLen_" + std::to_string(debye)+"_results.dat", std::ofstream::app);
  int sumCharge = 0;
  for(auto& p : Monomers)
  {
    if (p->GetCharge() < 0)
      sumCharge++;
  }

  double Rg_value = *((double*) lammps_extract_compute(lmp, "Rg_compute", 0, 0));
  double PE_value = *((double*) lammps_extract_compute(lmp, "myPE", 0, 0));
  double f_value = double(sumCharge)/double(Monomers.size());

  results_file<<f_value<<" "<<Rg_value<<" "<<PE_value<<std::endl;
  results_file.close();
}

//  WRITE THE LAMMPS DATA FILE
void WriteDataFile(LAMMPS* lmp, ParticleList &atoms, std::ofstream& ofs)
{

  int natoms = static_cast<int> (lmp->atom->natoms);
  double *vel = new double[3*natoms]; 
  int *image = new int[natoms];
  int *image_all = new int[3*natoms];
  lammps_gather_atoms(lmp, "image", 0, 1, image);
  lammps_gather_atoms(lmp, "v", 1, 3, vel);

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

      for(int i = 0; i < atoms.size(); i++)
      {
        ofs<<i+1<<" 1 "<<atoms[i]->GetSpeciesID()<<" "<<atoms[i]->GetCharge()<<" ";
        auto& xyz = atoms[i]->GetPosition();
        for(auto& x : xyz){
            ofs<<x<<" ";
        }
        ofs<<image_all[i*3]<<" "<<image_all[i*3+1]<<" "<<image_all[i*3+2];
        ofs<<std::endl;
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

    std::string vstr = "Velocities";
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
    }
    
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

void setSaphronBondedNeighbors(ParticleList &Monomers)
{
  int var0 = 0;
  int var1 = 0;
  int var2 = 0;
  int var3 = 0;

  cout <<"I am here TOO"<<endl;
  std::ifstream infile("data.trial");
  std::string line;

  int ii = 0;
  int store = 0;

  while (std::getline(infile, line))
  {
    if(line.empty())
      continue;

    std::istringstream iss(line);
    std::string s2 = iss.str();

    std::string s5 = "Bonds";
    if (s2.std::string::find(s5) != std::string::npos)
    {
      store = ii;
      break;
    }
    ii++;
  }

  int do_not_detect_2 = 0;
  while (std::getline(infile, line))
  {
    if(line.empty())
      continue;

    std::istringstream iss(line);

    if (ii>=store)
    {
      iss >> var0 >> var1 >> var2 >> var3;
      cout << var0 << var1 << var2 << var3 << "\n";
      cout <<"The line got is "<<line <<endl;

      if (var2 == 1)
      {
        Monomers[var2-1]->AddBondedNeighbor(Monomers[var3-1]);
        Monomers[var3-1]->AddBondedNeighbor(Monomers[var2-1]);
        Monomers[var3-1]->AddBondedNeighbor(Monomers[(var3+1)-1]);
        do_not_detect_2 = var3+1;

      }

      if (var2 != 1 && var3!=do_not_detect_2)
      {
        Monomers[var2-1]->AddBondedNeighbor(Monomers[var3-1]);
        Monomers[var2-1]->AddBondedNeighbor(Monomers[(var3-2)-1]);
      }

    }
  }

}