#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include <omp.h>
#include <time.h>
#include "../src/ForceFields/ForceFieldManager.h"
#include "../src/ForceFields/LennardJonesTSFF.h"
#include "../src/ForceFields/DSFFF.h"
#include "../src/ForceFields/FENEFF.h"
#include "../src/Moves/MoveManager.h"
#include "../src/Moves/Move.h"
#include "../src/Moves/InsertParticleMove.h"
#include "../src/Moves/DeleteParticleMove.h"
#include "../src/Moves/AcidReactionMove.h"
#include "../src/Moves/AnnealChargeMove.h"
#include "../src/Particles/Particle.h"
#include "../src/Worlds/World.h"
#include "../src/Worlds/WorldManager.h"
#include "../src/Properties/Energy.h"

#include "lammps.h"         // these are LAMMPS include files
#include "input.h"
#include "atom.h"
#include "library.h"

using namespace std;
using namespace SAPHRON;
using namespace LAMMPS_NS;

// forward declaration fkjldfdvfjkndfvjkndfvjkn
void WriteDataFile(int numatoms, ParticleList &atoms);
void saphronLoop(LAMMPS* &lmp, int &lammps, MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, ParticleList &Monomers, World &world); //const SAPHRON::MoveOverride &override
int main(int narg, char **arg)
{
// Set up SAPHRON in the main function fdeijvfnedilkvnkdferferfreerfr
  ParticleList Monomers;
  ForceFieldManager ffm;
  SAPHRON::Particle poly("Polymer");
  double rcut = 3.0;
  World world(10.38498, 10.38498, 10.38498, rcut, 46732);
  WorldManager WM;
  WM.AddWorld(&world);
  MoveManager MM (time(NULL));
  //MoveOverride override = SAPHRON::MoveOveride::None;
  //enum MoveOveride override = None;

  LennardJonesTSFF lj(0.8, 1.0, {1.122});
  FENEFF fene(1.0, 1.0, 30.0, 2.0);
  // Electrostatics
  // DSFFF dsf(0.1, 3.0);    
  // ffm.AddNonBondedForceField("Monomer", "Monomer", dsf);

  //ffm.AddNonBondedForceField("Monomer", "Monomer", lj);
  //ffm.AddBondedForceField("Monomer", "Monomer", fene);

  //InsertParticleMove Ins({{"Monomer"}}, WM,20,false,time(NULL));
  //DeleteParticleMove Del({{"Monomer"}},false,time(NULL));
  /*
          AcidReactionMove AcidMv({{"Monomer"}}, {{"temp"}},WM,20,10,time(NULL));*/
          AnnealChargeMove AnnMv({{"Polymer"}}, time(NULL));
  
  //MM.AddMove(&Ins);
  //MM.AddMove(&Del);
  /*
      MM.AddMove(&AcidMv);*/
     MM.AddMove(&AnnMv);
  


  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)

  //args[0] = mpistuff
  //args[1] = number processors lammps will use
  //args[2] = Lammps input file name

  MPI_Init(&narg,&arg);

  int me,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  int nprocs_lammps = atoi(arg[1]);

  if (nprocs_lammps > nprocs) {
    if (me == 0)
      printf("ERROR: LAMMPS cannot use more procs than available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  int lammps;
  if (me < nprocs_lammps) lammps = 1;
  else lammps = MPI_UNDEFINED;
  MPI_Comm comm_lammps;
  MPI_Comm_split(MPI_COMM_WORLD,lammps,0,&comm_lammps);
  
  // open LAMMPS input script
  FILE *fp;
  if (me == 0) {
    fp = fopen(arg[2],"r");
    if (fp == NULL) {
      printf("ERROR: Could not open LAMMPS input script\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  // First time lammps run (lammps equilibration run)
  // run the input script thru LAMMPS one line at a time until end-of-file
  // driver proc 0 reads a line, Bcasts it to all procs
  // (could just send it to proc 0 of comm_lammps and let it Bcast)
  // all LAMMPS procs call input->one() on the line
  
  // Create an instance of lammps and run for equilibration
  LAMMPS *lmp;
  if (lammps == 1) lmp = new LAMMPS(0,NULL,comm_lammps);
  else
  {
  	std::cout<<"Couldn't make LAMMPS instance!"<<std::endl;
  	exit(-1);
  }
  int n;
  char line[1024];
  while (1) 
  {
    if (me == 0) 
    {
      if (fgets(line,1024,fp) == NULL) n = 0;
      else n = strlen(line) + 1;
      if (n == 0) fclose(fp);
    }

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if (n == 0) break;
    MPI_Bcast(line,n,MPI_CHAR,0,MPI_COMM_WORLD);
    if (lammps == 1) lmp->input->one(line);

  }

  int natoms = static_cast<int> (lmp->atom->natoms);
  double *x = new double[3*natoms];
  lammps_gather_atoms(lmp,"x",1,3,x);
  Rand _rand(time(NULL));

  // Intialize monomers
  for(int i=0; i<natoms*3;i=i+3)
  {
    Monomers.push_back(new Particle({x[i],x[i+1],x[i+2]},{0.0,0.0,0.0}, "Monomer"));
  }

  for(int i=1; i<natoms-1;i++)
  {
    Monomers[i]->AddBondedNeighbor(Monomers[i+1]);
    Monomers[i]->AddBondedNeighbor(Monomers[i-1]);
  }

  Monomers[0]->AddBondedNeighbor(Monomers[1]);
  Monomers[natoms-1]->AddBondedNeighbor(Monomers[natoms-2]);

  for(auto& c : Monomers)
  {
    poly.AddChild(c);
  }
  ffm.AddNonBondedForceField("Monomer", "Monomer", lj);
  ffm.AddBondedForceField("Monomer", "Monomer", fene);
  world.AddParticle(&poly);
  world.SetChemicalPotential("Monomer", 10);

  delete [] x;

  //Make sure all threads are caught up to this point.
  MPI_Barrier(MPI_COMM_WORLD);

  // WHILE LOOP (alternating between saphron and lammps)
  int loop = 0;
  while(loop < 1)
  {
    // Run saphron for M steps. Includes energy evaluation and create a lammps data file within this function
    saphronLoop(lmp, lammps, MM, WM, ffm, Monomers, world); // SAPHRON::MoveOverride::None

    // Delete instance of LAMMPS
    if (lammps == 1) delete lmp;

    // Create lammps instance
      LAMMPS *lmp;
      if (lammps == 1) lmp = new LAMMPS(0,NULL,comm_lammps);

    // Read lammps input file (it will read the data file line also)
      FILE *fp;
      if (me == 0) {
        fp = fopen("in.sammps2","r");
        if (fp == NULL) {
          printf("ERROR: Could not open LAMMPS input script\n");
          MPI_Abort(MPI_COMM_WORLD,1);
        }
      }

      int n;
      char line[1024];
      while (1) 
      {
        if (me == 0) 
        {
          if (fgets(line,1024,fp) == NULL) n = 0;
          else n = strlen(line) + 1;
          if (n == 0) fclose(fp);
        }

        MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
        if (n == 0) break;
        MPI_Bcast(line,n,MPI_CHAR,0,MPI_COMM_WORLD);
        if (lammps == 1) lmp->input->one(line);
      }

     // Run lammps for N steps, lammps_loop function deleted
      lmp->input->one("run 10000"); // can be passed as an argument args[3] for example
      Rand _rand(time(NULL));
      loop++;
      cout << "the loop number is"<<loop<<endl;
  }
  if (lammps == 1) delete lmp;
  // close down MPI
  MPI_Finalize();
}


// FUNCTION
void saphronLoop(LAMMPS* &lmp, int &lammps, MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, ParticleList &Monomers, World &world){ // const SAPHRON::MoveOverride &override

      // Gather coordinates from lammps instance
      //if (lammps == 1)
      //{
        cout<<"I am here"<<endl;
        int natoms = static_cast<int> (lmp->atom->natoms);
        double *x = new double[3*natoms];
        lammps_gather_atoms(lmp,"x",1,3,x);
        Rand _rand(time(NULL));
      //}

      // Set position of monomers 
      int j = 0;
      for(int i=0; i<natoms*3;i=i+3){
            Monomers[j]->SetPosition({x[i],x[i+1],x[i+2]});
            j++;
      }

      // Energy evaluation and setting
      auto H1 = ffm.EvaluateEnergy(world);
      world.SetEnergy(H1.energy);
      world.UpdateNeighborList();
      delete [] x;

      // Perform moves for M steps ()
      for(int i=0; i<10;i++)
      {
        cout<<"I am here too"<<endl;
            // picks either insert or delete
        auto* move = MM.SelectRandomMove();
        move->Perform(&WM,&ffm,MoveOverride::None);
      }

      //Write out datafile that is utilized by lammps input script
      WriteDataFile(natoms, Monomers);
}


//  WRITE THE LAMMPS DATA FILE
void WriteDataFile(int numatoms, ParticleList &atoms)
{
  cout<<"how come i am here"<< endl;
  std::ofstream ofs;
  ofs.open ("Vik_Smells.dat", std::ofstream::out); // Vik_Smells.dat
  int lammps_atoms = 0;

  //Read in file and change what is needed
  std::ifstream infile("4_LJ_atoms.chain");
  std::string line;
  while (std::getline(infile, line))
  {
      if(line.empty())
        continue;
      std::istringstream iss(line);
      int a;
      std::string b;
      if ((iss >> a >> b))
      {
        cout<<"this is flag"<<endl;
        if(b == "atoms")
        {
          cout<<"b is equal to atoms"<<endl;
          lammps_atoms = a;
          a = numatoms;
          //line = std::to_string(a)+" atoms";
          ofs<<"       "<<std::to_string(a)<<" atoms"<<std::endl;
          continue;
        }
        /*if(b == "Masses")
        {
          //line = std::to_string(a)+" atoms";
          ofs<<"\r\n";
          ofs<<"     "<<"1"<<" 1.0"<<std::endl;
          continue;
        }*/
      }
      else if((iss >> b))
      {
        cout<<"wow there horsie"<<std::endl;
        
        if( b == "Velocities")
        {
          int i = 0;
          while(i < lammps_atoms)
          {
            std::getline(infile,line);
            if(line.empty())
              continue;
            i++;
          }
          continue;
        }
        else if ( b == "Atoms")
        {
          ofs<<"Atoms"<<std::endl;
          ofs<<std::endl;
          cout<<"what is this"<<std::endl;
          int i = 1;
          for(auto& p : atoms)
          {
            ofs<<i<<" 0 "<<p->GetSpeciesID()<<" "<<p->GetCharge()<<" ";
            auto& xyz = p->GetPosition();
            int stop = 0;
            for(auto& x : xyz){
              if (stop<3)
              {
                ofs<<x<<" ";
              }
              stop++;
            }
            ofs<<std::endl;
            i++;
          }
          continue;
        }
       /* else if (b == "Bonds")
        {
          ofs<<"\r\n";
          continue;
        }*/
      }

      ofs<<line<<std::endl;
  }
}

//1 molecule-tag atom-type q x y z   (FOR ATOM STYLE FULL)