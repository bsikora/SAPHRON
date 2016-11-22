#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "stdio.h"
#include <math.h> 
#include <cmath> 
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
#include "../src/Moves/SpeciesSwapMove.h"
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
void ReadInputFile(std::string lammpsfile, MPI_Comm& comm_lammps, LAMMPS* lmp, int& xrand);
void SAPHRONLoop(MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, World &world);
void WriteDataFile(LAMMPS* lmp, World &world, std::ofstream& data_file, double box, int totalatoms);
void WriteResults(World &world, std::ofstream& results_file, double &debye, double &mu_NaCl_calc); // erkjnfjvkjk
void WriteDump(LAMMPS* lmp, World &world, std::ofstream& dump_file, double &debye, int &loop); // erdjkfdkjjkdfjnk

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
  double box = atof(arg[7]);
  int numNa = atof(arg[8]);
  int numOH = atof(arg[9]);
  int numCl = atof(arg[10]);
  numCl = numOH + numCl;

  double muNa_excess = atof(arg[11]);
  double muOH_excess = atof(arg[12]);
  double muCl_excess = atof(arg[13]);

  double muNa = (-log(box*box*box/numNa) + muNa_excess); // multiply by 0.5
  double muOH = -log(box*box*box/numOH) + muOH_excess;
  double muCl = (-log(box*box*box/numCl) + muCl_excess); // multiply by 0.5

  // this is the chemical potential of whole salt
  double box_volume = box*box*box;
  double mu_NaCl_calc = 1.0*log(pow(numNa/box_volume, 0.5) * pow(numCl/box_volume, 0.5)) + 0.5*muNa_excess + 0.5*muCl_excess;

  // taking into account minimum image convention
  if (coulcut > (box/2))
  {
    coulcut = box/2;
  }

  std::ofstream dump_file;
  dump_file.open("dump_debyeLen_" + std::to_string(debye)+"_.dat", std::ofstream::out);
  dump_file<<"id   type   q   x   y   z   ix   iy   iz"<<std::endl;
  dump_file.close();
  std::ofstream results_file;
  results_file.open("New_chemical_pot" + std::to_string(debye)+"_results.dat", std::ofstream::out);
  results_file<<"Na    cl    mu_salt_saph    mu_NaCl_calc"<<std::endl;
  results_file.close();

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
  std::string ions_input_file = "in.just_OH_and_salt";
  ReadInputFile(ions_input_file, comm_lammps, lmp, xrand);
  
  string random;
  char line[1024];

  int natoms = static_cast<int> (lmp->atom->natoms);
  double *x = new double[3*natoms];
  int *type = new int[natoms];
  lammps_gather_atoms(lmp, "x", 1, 3, x);
  lammps_gather_atoms(lmp,"type", 0, 1, type);

  /////////////////SETUP SAPHRON//////////////////////////////////////////////////////////////////////
  World world(box, box, box, coulcut, seed + 1); // same as lammps input data file
  world.SetTemperature(1.0);

  SAPHRON::Particle* sodium = new Particle({10.0,0.0,0.0},{0.0,0.0,0.0}, "Sodium"); // set pointers
  SAPHRON::Particle* chloride = new Particle({100.0,0.0,0.0},{0.0,0.0,0.0}, "Chloride");

  sodium->SetCharge(1.67); // dereference the pointer and set the charge
  chloride->SetCharge(-1.67);

  for(int i=0; i<natoms*3; i=i+3)
  {
    if(type[i/3] == 3)
    {
      Particle* pnew = sodium->Clone();
      pnew->SetPosition({x[i],x[i+1],x[i+2]});
      world.AddParticle(pnew);
    }
    else if(type[i/3] == 2 || type[i/3] == 1)
    {
      Particle* pnew = chloride->Clone();
      pnew->SetPosition({x[i],x[i+1],x[i+2]});
      world.AddParticle(pnew);
    }
  }

  //Create world
  WorldManager WM;
  WM.AddWorld(&world);
  world.SetChemicalPotential("Sodium", muNa);
  world.SetChemicalPotential("Chloride", muCl); // changed here

  //Create forcefields
  ForceFieldManager ffm;
  //LennardJonesTSFF lj(1.0, 1.0, {2.5}); // Epsilon, sigma, cutoff
  LennardJonesTSFF lj2(1.0, 1.0, {1.122}); // Epsilon, sigma, cutoff
  //FENEFF fene(0, 1.0, 7.0, 2.0); // epsilon, sigma, kspring, rmax
  DSFFF DSF(0, {coulcut});
  ffm.SetElectrostaticForcefield(DSF);

  ffm.AddNonBondedForceField("Chloride", "Chloride", lj2);
  ffm.AddNonBondedForceField("Chloride", "Sodium", lj2);
  ffm.AddNonBondedForceField("Sodium", "Sodium", lj2);

  //Set up moves
  MoveManager MM (seed);
  InsertParticleMove Ins1({{"Sodium"}, {"Chloride"}}, WM, 20, true, seed + 3);
  DeleteParticleMove Del1({{"Sodium"}, {"Chloride"}}, true, seed + 4);

  MM.AddMove(&Ins1);
  MM.AddMove(&Del1);

  int ta = 0;
  for(const auto& p : world)
  {
    if(p->HasChildren())
    {
      for(auto& pp : p->GetChildren())
      {
        ta++;
      }
    }
    else
    {
      ta++;
    }
  }

  std::ofstream data_file;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
    data_file.open("data."+lammpsfile, std::ofstream::out);
  WriteDataFile(lmp, world, data_file, box, ta);
  if (rank == 0)
    data_file.close();

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
    cout<<"Read input"<<endl;
    ReadInputFile(lammpsfile, comm_lammps, lmp, xrand);
    natoms = static_cast<int> (lmp->atom->natoms);
    double *x = new double[3*natoms];
    lammps_gather_atoms(lmp,"x",1,3,x);

    // Set position of monomers
    int i = 0;
    for(const auto& p : world)
    {
      if(p->HasChildren())
      {
        for(auto& pp : p->GetChildren())
        {
          pp->SetPosition({x[i],x[i+1],x[i+2]});
          i=i+3;
        }
      }
      else
      {
        p->SetPosition({x[i],x[i+1],x[i+2]});
        i=i+3;
      }
    }

    SAPHRONLoop(MM, WM, ffm, world);

    int totalatoms = world.GetPrimitiveCount();

    if (rank == 0)
      data_file.open("data."+lammpsfile, std::ofstream::out);
    WriteDataFile(lmp, world, data_file, box, totalatoms);
    if (rank == 0)
      data_file.close();
    WriteResults(world, results_file, debye, mu_NaCl_calc);
    if (hit_detection_numb == 9)
    {
    	WriteDump(lmp, world, dump_file, debye, loop);
    	hit_detection_numb = -1;
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
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
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
  for(int i=0; i<18; i++)
  {
    auto* move = MM.SelectRandomMove();
    move->Perform(&WM, &ffm, MoveOverride::None);
  }    
}

void WriteResults(World &world, std::ofstream& results_file, double &debye, double &mu_NaCl_calc)
{
  results_file.open("New_chemical_pot" + std::to_string(debye)+"_results.dat", std::ofstream::app);
  const double mu_Na = world.GetChemicalPotential("Sodium");
  const double mu_Cl = world.GetChemicalPotential("Chloride");
  double mu_salt_saph = mu_Na + mu_Cl;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
  {
    results_file<<mu_Na<<" "<<mu_Cl<<" "<<mu_salt_saph<<" "<<mu_NaCl_calc<<std::endl;
    results_file.close();
  }
}

//  WRITE THE LAMMPS DATA FILE
void WriteDataFile(LAMMPS* lmp, World &world, std::ofstream& ofs, double box, int s_natoms)
{

  int natoms = static_cast<int> (lmp->atom->natoms);
  double *vel = new double[3*natoms]; 
  int *image = new int[natoms];
  int *image_all = new int[3*natoms];
  int *type = new int[natoms];
  lammps_gather_atoms(lmp, "image", 0, 1, image);
  lammps_gather_atoms(lmp, "v", 1, 3, vel);
  lammps_gather_atoms(lmp,"type", 0, 1, type);

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
    // THIS TIME READING IS NOT DATA.TRIAL BUT SOME OTHER FILE that will be template
    std::ifstream infile("data.template");
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
        ofs<<"       "<<std::to_string(s_natoms)<<" atoms"<<std::endl;
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
          else
          {
            ofs<<i+1<<" 1 "<<(p->GetSpeciesID()+1)<<" "<<p->GetCharge()<<" ";
            for(auto& x : p->GetPosition()){
                ofs<<x<<" ";
            }
            ofs<<"0 0 0"<<std::endl;
            i++;
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

      std::string s6 = "xlo";
      if (s2.std::string::find(s6) != std::string::npos)
      {
        ofs<<"       0.0 "<< box <<" xlo xhi"<<std::endl;
        continue;
      }

      std::string s7 = "ylo";
      if (s2.std::string::find(s7) != std::string::npos)
      {
        ofs<<"       0.0 "<< box <<" ylo yhi"<<std::endl;
        continue;
      }

      std::string s8 = "zlo";
      if (s2.std::string::find(s8) != std::string::npos)
      {
        ofs<<"       0.0 "<< box <<" zlo zhi"<<std::endl;
        continue;
      }

      ofs<<iss.str()<<std::endl;
    }
    ofs.close();
  }

  delete [] image;
  delete [] image_all;
  delete [] vel;
  delete [] type;
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

  int MDstep = loop*1000;
  dump_file.open("dump_debyeLen_" + std::to_string(debye)+"_.dat", std::ofstream::app);
  dump_file<<"The MD step is: "<<MDstep<<std::endl;

  int k = 0;
  for(const auto& p : world)
  {
    dump_file<<k+1<<" "<<p->GetSpeciesID()<<" "<<p->GetCharge()<<" ";
    for(auto& x : p->GetPosition()){
        dump_file<<x<<" ";
    }
    dump_file<<"0 0 0"<<std::endl;
    k++;    
  }
  dump_file.close();

  delete [] image;
  delete [] image_all; 
}
