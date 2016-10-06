#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "stdio.h"
#include <math.h> 
#include "stdlib.h"
#include "string.h"
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
void WriteCodedRgAnalysisFile(vector<double>& rgVec, char* arg);
double calculateRg(LAMMPS* &lmp);
void WriteDataFile(int numatoms, ParticleList &atoms, int* img, char* arg, int &loop);
void WriteFractionAnalysisFile(vector<double>& chgVec, char* arg);
void readInputFile(LAMMPS* &lmp, std::string &inFile);
void WriteRgAnalysisFile(vector<double>& rgVec, char* arg);
void WritePEAnalysisFile(vector<double>& peVec, char* arg);
void saphronLoop(LAMMPS* &lmp, int &lammps, MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, ParticleList &Monomers, World &world, vector<double>& chgVec, char* arg, int &loop); //const SAPHRON::MoveOverride &override

// Main code
int main(int narg, char **arg)
{

  std::string yol = "in.polymer_new2_debLen_"+std::string(arg[7]); 
  std::vector<double> chargeVector;
  std::vector<double> rgVector;
  std::vector<double> codedrgVector;
  std::vector<double> peVector;
  ParticleList Monomers;
  ForceFieldManager ffm;
  SAPHRON::Particle poly("Polymer");
  double rcut = 300.0;
  World world(1000.0, 1000.0, 1000.0, rcut, 46732); // same as lammps input data file
  world.SetTemperature(1.0);
  WorldManager WM;
  WM.AddWorld(&world);
  MoveManager MM (time(NULL));

  LennardJonesTSFF lj(1.0, 1.0, {2.5});
  FENEFF fene(0, 1.0, 7.0, 2.0);
  DebyeHuckelFF debHuc(atof(arg[4]), {atof(arg[5])}); // same as lammps input ;  kappa (1/deb len), coul cutoff (5*deb len)


  //InsertParticleMove Ins({{"Monomer"}}, WM,20,false,time(NULL));
  //DeleteParticleMove Del({{"Monomer"}},false,time(NULL));
  //AcidReactionMove AcidMv({{"Monomer"}}, {{"temp"}},WM,20,10,time(NULL));

  AnnealChargeMove AnnMv({{"Polymer"}}, time(NULL));
  MM.AddMove(&AnnMv);

  //MM.AddMove(&Ins);
  //MM.AddMove(&Del);
  //MM.AddMove(&AcidMv);
  
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
  LAMMPS *Oldlmp;
  LAMMPS *Newlmp;

  if (lammps == 1) Oldlmp = new LAMMPS(0,NULL,comm_lammps);
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
    Oldlmp->input->one(line);

  }

  // ******************************INITIALIZATION**************************************//
  int natoms = static_cast<int> (Oldlmp->atom->natoms);
  double *x = new double[3*natoms];
  lammps_gather_atoms(Oldlmp,"x",1,3,x);
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

  // Making each monomer child of polymer
  for(auto& c : Monomers)
  {
    poly.AddChild(c);
  }

  ffm.SetElectrostaticForcefield(debHuc);
  ffm.AddNonBondedForceField("Monomer", "Monomer", lj);

  ffm.AddBondedForceField("Monomer", "Monomer", lj); // changed bonded force field
  ffm.AddBondedForceField("Monomer", "Monomer", fene); // changed bonded force field
  ffm.AddBondedForceField("Monomer", "Monomer", debHuc); // changed bonded force field

  world.AddParticle(&poly);

  // Adding titration moves
  // proton charge based on bjerrum length 2.8sigma (e*sqrt(2.8) == 1.67)
  AcidTitrationMove AcidTitMv({{"Monomer"}}, 1.67, atof(arg[6]), time(NULL));  // proton charge, mu
  MM.AddMove(&AcidTitMv);

  cout<<"the mu is "<< atof(arg[6])<<endl;

  delete [] x;

  //Make sure all threads are caught up to this point.
  MPI_Barrier(MPI_COMM_WORLD);
/*####################$$$$$$$$$$$$$$@@@@@@@@@@@@@@@@@@@@@************%%%%%%%%%%%%%%%%%%%%%%%%#&&&&&&&&&&&&&&&&&&&*/




  std::string::size_type sz;
  int numLoops = std::stoi(arg[3],&sz);
  int loop = 0;
  // WHILE LOOP (alternating between saphron and lammps)
  while(loop < numLoops)   
  {
    // Run saphron. Includes energy evaluation and create a lammps data file within this function
    if(loop == 0)
    {
      saphronLoop(Oldlmp, lammps, MM, WM, ffm, Monomers, world, chargeVector, arg[7], loop); // SAPHRON::MoveOverride::None
      delete Oldlmp;
    }
    else
    {
      saphronLoop(Newlmp, lammps, MM, WM, ffm, Monomers, world, chargeVector, arg[7], loop); // SAPHRON::MoveOverride::None
      delete Newlmp;
    }

    Newlmp = new LAMMPS(0,NULL,comm_lammps);
    readInputFile(Newlmp, yol);

    Newlmp->input->one("run 1000");
    Rand _rand(time(NULL));

    double Rg_value = *((double*) lammps_extract_compute(Newlmp,"Rg_compute",0,0));
    double PE_value = *((double*) lammps_extract_compute(Newlmp,"myPE",0,0));
    double coded_Rg_value = calculateRg(Newlmp);
	cout<< "the coded Rg is "<< coded_Rg_value<<endl;

    codedrgVector.push_back(coded_Rg_value);
    rgVector.push_back(Rg_value);
    peVector.push_back(PE_value);

    cout << "the loop number is "<<loop<<endl;
    loop++;
    
    WriteFractionAnalysisFile(chargeVector, arg[7]);
    WriteRgAnalysisFile(rgVector, arg[7]);
    WriteCodedRgAnalysisFile(codedrgVector, arg[7]);
    WritePEAnalysisFile(peVector, arg[7]);

    
  }
  Newlmp->input->one("log log.saph_loop_debLen_0.01");
  Newlmp->input->one("run 500000");
  delete Newlmp;


  
  // JUST PRINTING FOR CONFIRMATION
  double d = atof(arg[7]);
  double e = atof(arg[4]);
  double f = atof(arg[5]);
  double g = atof(arg[6]);
  cout<<"the debye len is "<< d<<endl;
  cout<<"the kappa is "<< e<<endl;
  cout<<"the coulCut is "<< f<<endl;
  cout<<"the mu is "<< g<<endl;

  // close down MPI
  MPI_Finalize();
}


// FUNCTION
void saphronLoop(LAMMPS* &lmp, int &lammps, MoveManager &MM, WorldManager &WM, ForceFieldManager &ffm, ParticleList &Monomers, World &world, vector<double>& chgVec, char* arg, int &loop){ // const SAPHRON::MoveOverride &override

	    cout<<"I am in saphron loop here"<<endl;
	    int natoms = static_cast<int> (lmp->atom->natoms);
	    cout << "the number of atoms is " << natoms << endl;
	    double *x = new double[3*natoms];
	    double *q = new double[natoms];
      int *image = new int[natoms];
      int *image_all = new int[3*natoms];
	    lammps_gather_atoms(lmp,"x",1,3,x);
	    lammps_gather_atoms(lmp,"q",1,1,q);
      lammps_gather_atoms(lmp,"image",0,1,image);
	    Rand _rand(time(NULL));

      for (int i = 0; i < natoms; i++)
      {
        image_all[i*3] = (image[i] & IMGMASK) - IMGMAX;;
        image_all[i*3+1] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        image_all[i*3+2] = (image[i] >> IMG2BITS) - IMGMAX;
      }


      // Set position of monomers 
      int j = 0;
      for(int i=0; i<natoms*3;i=i+3){
            Monomers[j]->SetPosition({x[i],x[i+1],x[i+2]});
            j++;
      }

      // Set charges on the monomers
      int k = 0;
      for(auto& chg : Monomers)
      {
      	chg->SetCharge(q[k]);
      	k++;
      }

      // Energy evaluation and setting
      auto H1 = ffm.EvaluateEnergy(world);
      world.SetEnergy(H1.energy);
      world.UpdateNeighborList();
      delete [] x;


      // Perform moves for M steps ()
      for(int i=0; i<30;i++)
      {
        auto* move = MM.SelectRandomMove();
        move->Perform(&WM,&ffm,MoveOverride::None); // performs all the moves like dabbing
      }


    // FRACTION OF CHARGE CALCULATION
    int intCharge = 0;
    int intMonomers = 0;
    for(auto& p : Monomers)
      {
      	intMonomers++;
      	if (p->GetCharge())
      	{
      		intCharge++;
      	}
      }
      chgVec.push_back((double)intCharge/intMonomers);

      //Write out datafile that is utilized by lammps input script
      WriteDataFile(natoms, Monomers, image_all, arg, loop);    
}


//  WRITE THE LAMMPS DATA FILE
void WriteDataFile(int numatoms, ParticleList &atoms, int* img, char* arg, int &loop)
{
  std::ofstream ofs;
  ofs.open ("data.polymer_debLen_"+std::string(arg), std::ofstream::out);
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
      ofs<<"       "<<std::to_string(numatoms)<<" atoms"<<std::endl;
      iss >> numlammpsatoms >> garbage;
      std::cout<<"Num lammps atoms is "<<numlammpsatoms<<std::endl;
      continue;
    }
    std::string s3 = "Atoms";
    if (s2.std::string::find(s3) != std::string::npos)
    {
      ofs<<"Atoms"<<std::endl;
      ofs<<std::endl;
      int i = 1;
      int ii = 0;
      int jj = 0;
      while(ii < numlammpsatoms) // this while loop is important
        // this makes sure that you don't read lines corresponding to previous lammps coordinates
      {
        std::getline(infile,line);
        if(line.empty())
          continue;
        ii++;
      }

      for(auto& p : atoms)
      {
        ofs<<i<<" 1 "<<p->GetSpeciesID()<<" "<<p->GetCharge()<<" ";
        auto& xyz = p->GetPosition();
        for(auto& x : xyz){
            ofs<<x<<" ";
        }
        ofs<<img[jj]<<" "<<img[jj+1]<<" "<<img[jj+2];
        ofs<<std::endl;
        i++;
        jj+=3;
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
      ofs<<"       "<<std::to_string(0.0)<<" "<<std::to_string(1000.0)<<" xlo xhi"<<std::endl;
      continue;
    }

    std::string s7 = "ylo";
    if (s2.std::string::find(s7) != std::string::npos)
    {
      ofs<<"       "<<std::to_string(0.0)<<" "<<std::to_string(1000.0)<<" ylo yhi"<<std::endl;
      continue;
    }

    std::string s8 = "zlo";
    if (s2.std::string::find(s8) != std::string::npos)
    {
      ofs<<"       "<<std::to_string(0.0)<<" "<<std::to_string(1000.0)<<" zlo zhi"<<std::endl;
      continue;
    }

    ofs<<iss.str()<<std::endl;
  }
  ofs.close();


  if (loop == 5000 || loop == 15000 || loop == 20999 || loop == 40000 || loop == 49999)
  {
  	  std::ofstream ofs2;
  	  ofs2.open ("copy_data_file_debyeLen_"+std::string(arg)+"_"+std::to_string(loop)+".dat", std::ofstream::out);
      std::ifstream infile2("data.polymer_debLen_"+std::string(arg));
	  while (std::getline(infile2, line))
	  {
	    //if(line.empty())
	      //continue;

	    std::istringstream iss(line);
	    ofs2<<iss.str()<<std::endl;
	  }
  }

}

void WriteFractionAnalysisFile(vector<double>& chgVec, char* arg)
{
	 std::ofstream ofs;
     ofs.open ("fraction_charged_debyeLen_"+std::string(arg)+".dat", std::ofstream::out);
    for (std::vector<double>::iterator it = chgVec.begin() ; it != chgVec.end(); ++it)
    {
    	ofs<<std::to_string(*it)<<std::endl;
    }
}

void WriteRgAnalysisFile(vector<double>& rgVec, char* arg)
{
   std::ofstream ofs;
     ofs.open ("Rg_debyeLen_"+std::string(arg)+".dat", std::ofstream::out);
    for (std::vector<double>::iterator it = rgVec.begin() ; it != rgVec.end(); ++it)
    {
      ofs<<std::to_string(*it)<<std::endl;
    }
}

void WriteCodedRgAnalysisFile(vector<double>& rgVec, char* arg)
{
   std::ofstream ofs;
     ofs.open ("Coded_Rg_debyeLen_"+std::string(arg)+".dat", std::ofstream::out);
    for (std::vector<double>::iterator it = rgVec.begin() ; it != rgVec.end(); ++it)
    {
      ofs<<std::to_string(*it)<<std::endl;
    }
}

void WritePEAnalysisFile(vector<double>& peVec, char* arg)
{
   std::ofstream ofs;
     ofs.open ("PE_debyeLen_"+std::string(arg)+".dat", std::ofstream::out);
    for (std::vector<double>::iterator it = peVec.begin() ; it != peVec.end(); ++it)
    {
      ofs<<std::to_string(*it)<<std::endl;
    }
}

void readInputFile(LAMMPS* &lmp, std::string &inFile)   //int &me, 
{
    FILE *fp;
    //if (me == 0) {
      fp = fopen(inFile.c_str(),"r");
      cout<<"file name passed to readInputFile "<<inFile.c_str()<<endl;
      if (fp == NULL) {
        printf("ERROR: Could not open LAMMPS input script\n");
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    //}

    int n;
    char line[1024];
    while (1) 
    {
      //if (me == 0) 
      //{
        if (fgets(line,1024,fp) == NULL) n = 0;
        else n = strlen(line) + 1;
        if (n == 0) fclose(fp);
      //}

      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
      if (n == 0) break;
      MPI_Bcast(line,n,MPI_CHAR,0,MPI_COMM_WORLD);
      lmp->input->one(line);
    }
}

double calculateRg(LAMMPS* &lmp)
{
      int natoms = static_cast<int> (lmp->atom->natoms);

      double *x = new double[3*natoms];
      double *x_unwrapped = new double[3*natoms];
      double *diff_r_rcm = new double[3*natoms];
      double *dot_prod_array = new double[natoms];
      int *image = new int[natoms];
      int *image_all = new int[3*natoms];

      lammps_gather_atoms(lmp,"x",1,3,x);
      lammps_gather_atoms(lmp,"image",0,1,image);
      Rand _rand(time(NULL));

      // GET ALL THE IMAGE FLAGS IN A 1D ARRAY
      for (int i = 0; i < natoms; i++)
      {
        image_all[i*3] = (image[i] & IMGMASK) - IMGMAX;;
        image_all[i*3+1] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        image_all[i*3+2] = (image[i] >> IMG2BITS) - IMGMAX;
      }


      // CENTER OF MASS CALCULATION
      double x_sum = 0;
      double y_sum = 0;
      double z_sum = 0;

      double x_com = 0;
      double y_com = 0;
      double z_com = 0;

      double tot_mass = natoms; // since mass 1
      double box_length = 1000; // since cubic box

      for(int i=0; i<natoms*3;i=i+3){
            x_sum += x[i] + image_all[i]*box_length;
            y_sum += x[i+1] + image_all[i+1]*box_length;
            z_sum += x[i+2] + image_all[i+2]*box_length;
      }

      x_com = x_sum/tot_mass;
      y_com = y_sum/tot_mass;
      z_com = z_sum/tot_mass;


      // DOT PRODUCT CALCULATION
      int j = 0;
      for(int i=0; i<natoms*3;i=i+3){
            x_unwrapped[i] = x[i] + image_all[i]*box_length;
            x_unwrapped[i+1] = x[i+1] + image_all[i+1]*box_length;
            x_unwrapped[i+2] = x[i+2] + image_all[i+2]*box_length;

            diff_r_rcm[i] = x_unwrapped[i] - x_com;
            diff_r_rcm[i+1] = x_unwrapped[i+1] - y_com;
            diff_r_rcm[i+2] = x_unwrapped[i+2] - z_com;

            dot_prod_array[j] = diff_r_rcm[i]*diff_r_rcm[i] + diff_r_rcm[i+1]*diff_r_rcm[i+1] + diff_r_rcm[i+2]*diff_r_rcm[i+2];
            j++;
      }


      // FINAL RG CALCULATION
      double Rg_summation = 0;
      double Rg = 0;
      for (int i = 0; i < natoms; ++i)
      {
      	Rg_summation += dot_prod_array[i];
      }

      Rg = sqrt(Rg_summation/tot_mass);

      return Rg;  
}

//1 molecule-tag atom-type q x y z   (FOR ATOM STYLE FULL)
