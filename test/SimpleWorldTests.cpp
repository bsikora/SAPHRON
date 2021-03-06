#include "../src/Particles/Particle.h"
#include "../src/Worlds/World.h"
#include "gtest/gtest.h"
#include <map>
#include <chrono>
#include <ctime>

using namespace SAPHRON;

TEST(SimpleWorld, WorldProps)
{
	// Pack the world.
	World world(1, 1, 1, 1.0, 1.0);
	Particle site1({0, 0, 0}, {1, 0, 0}, "E1");
	Particle site2({0, 0, 0}, {0, 1, 0}, "E2");
	Particle site3({0, 0, 0}, {0, 0, 1}, "E3");

	// Pack the world with 3 species.
	world.PackWorld({&site1, &site2, &site3}, 
					{1.0/3.0, 1.0/3.0, 1.0/3.0}, 
					4500, 
					0.5);

	ASSERT_EQ(4500, world.GetParticleCount());
	ASSERT_EQ(4500, world.GetPrimitiveCount());

	// Stash some particles. 
	world.StashParticle(&site1, 100);

	int s1 = site1.GetSpeciesID();
	int s2 = site2.GetSpeciesID();
	int s3 = site3.GetSpeciesID();

	// Unstash and add to world then check counts.
	// We expect stash to autofill itself when empty.
	for(int i = 0; i < 200; ++i)
		world.AddParticle(world.UnstashParticle(s1));

	auto& comp = world.GetComposition();

	ASSERT_EQ(1700, comp[s1]);

	// Test chemical potential. 
	ASSERT_EQ(0, world.GetChemicalPotential(s1));
	ASSERT_EQ(0, world.GetChemicalPotential(s2));
	ASSERT_EQ(0, world.GetChemicalPotential(s3));
	ASSERT_EQ(0, world.GetChemicalPotential("E1"));
	ASSERT_EQ(0, world.GetChemicalPotential("E2"));
	ASSERT_EQ(0, world.GetChemicalPotential("E3"));

	world.SetChemicalPotential("E2", 5.0);
	ASSERT_EQ(0.0, world.GetChemicalPotential(s1));
	ASSERT_EQ(5.0, world.GetChemicalPotential(s2));
	ASSERT_EQ(0.0, world.GetChemicalPotential(s3));

	world.SetChemicalPotential(s2, 6.0);
	ASSERT_EQ(0.0, world.GetChemicalPotential(s1));
	ASSERT_EQ(6.0, world.GetChemicalPotential(s2));
	ASSERT_EQ(0.0, world.GetChemicalPotential(s3));

	world.SetChemicalPotential(s3, -4.4);
	ASSERT_EQ(0.0, world.GetChemicalPotential(s1));
	ASSERT_EQ(6.0, world.GetChemicalPotential(s2));
	ASSERT_EQ(-4.4, world.GetChemicalPotential(s3));

	// Check de Broglie wavelength. 
	// In reduced units these should all be 1.
	ASSERT_EQ(1.0, world.GetWavelength(s1));
	ASSERT_EQ(1.0, world.GetWavelength(s2));
	ASSERT_EQ(1.0, world.GetWavelength(s3));
}

TEST(SimpleWorld, DrawParticlesBySpecies)
{
	World world(1, 1, 1, 1.0, 1.0);
	Particle site1({0, 0, 0}, {1, 0, 0}, "E1");
	Particle site2({0, 0, 0}, {0, 1, 0}, "E2");
	Particle site3({0, 0, 0}, {0, 0, 1}, "E3");

	// Pack the world with 3 species.
	world.PackWorld({&site1, &site2, &site3}, 
					{1.0/3.0, 1.0/3.0, 1.0/3.0}, 
					4500, 
					0.5);

	ASSERT_EQ(4500, world.GetParticleCount());
	ASSERT_EQ(4500, world.GetPrimitiveCount());

	int s1 = site1.GetSpeciesID();
	int s2 = site2.GetSpeciesID();
	int s3 = site3.GetSpeciesID();

	ASSERT_EQ(world.SelectParticle(0), world.SelectParticleBySpecies(s1, 0));
	ASSERT_EQ(world.SelectParticle(1500), world.SelectParticleBySpecies(s2, 0));
	ASSERT_EQ(world.SelectParticle(3000), world.SelectParticleBySpecies(s3, 0));
	ASSERT_EQ(world.SelectParticle(3004), world.SelectParticleBySpecies(s3, 4));
	ASSERT_EQ(nullptr, world.DrawRandomParticleBySpecies(44));

	for(int i = 0; i < 10000; ++i)
		ASSERT_EQ(s2, world.DrawRandomParticleBySpecies(s2)->GetSpeciesID());
}

TEST(SimpleWorld, NeighborList)
{
	// create world with specified nlist. 
	double rcut = 1.5;
	World world(30, 30, 30, 1.3*rcut, 0.3*rcut);
	ASSERT_DOUBLE_EQ(1.3*rcut, world.GetNeighborRadius());
	ASSERT_DOUBLE_EQ(0.3*rcut, world.GetSkinThickness());

	// Change values and make sure it propogates. 
	world.SetNeighborRadius(0.9);
	ASSERT_DOUBLE_EQ(0.9, world.GetNeighborRadius());
	ASSERT_DOUBLE_EQ(0.3*rcut, world.GetSkinThickness());

	// Pack a world with a decent number of of particles.
	Particle site1({0, 0, 0}, {1, 0, 0}, "E1");
	world.PackWorld({&site1}, {1.0}, 5000, 0.5);


    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    for(int i = 0; i < 100; ++i)
    	world.UpdateNeighborList();

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
}

TEST(SimpleWorld, DefaultBehavior)
{
	int n = 30;
	World world(n, n, n, 1.0, 1.0);
	Particle site1({0, 0, 0}, {1, 0, 0}, "E1");
	Particle site2({0, 0, 0}, {0, 1, 0}, "E2");
	Particle site3({0, 0, 0}, {0, 0, 1}, "E3");

	world.PackWorld({&site1, &site2, &site3}, {1.0/3.0, 1.0/3.0, 1.0/3.0});

	ASSERT_EQ(27000, world.GetParticleCount());
	ASSERT_EQ(27000, world.GetPrimitiveCount());


	std::map<std::string, int> counts {{"E1", 0}, {"E2", 0}, {"E3", 0}};
	for(int i = 0; i < world.GetParticleCount(); i++)
	{
		auto particle = world.SelectParticle(i);
		counts[particle->GetSpecies()]++;
	}

	ASSERT_EQ(9000, counts["E1"]);
	ASSERT_EQ(9000, counts["E2"]);
	ASSERT_EQ(9000, counts["E3"]);
	
	world.UpdateNeighborList();
	for(int i = 0; i < world.GetParticleCount(); ++i)
	{
		auto* particle = world.SelectParticle(i);
		const Position& coords = particle->GetPosition();

		// Check neighbors
		Position n1 =
		{(coords[0] == n) ? 1.0 : (double) coords[0] + 1.0, (double) coords[1], (double) coords[2]};
		Position n2 =
		{(coords[0] == 1) ? (double) n : coords[0] - 1.0, (double) coords[1], (double) coords[2]};
		Position n3 =
		{(double) coords[0], (coords[1] == n) ? 1.0 : (double) coords[1] + 1.0, (double) coords[2]};
		Position n4 =
		{(double) coords[0], (coords[1] == 1) ? (double) n : coords[1] - 1.0, (double) coords[2]};
		Position n5 =
		{(double) coords[0], (double) coords[1], (coords[2] == n) ? 1.0 : (double) coords[2] + 1.0};
		Position n6 =
		{(double) coords[0], (double) coords[1], (coords[2] == 1) ? (double) n : coords[2] - 1.0};

		auto& neighbors = particle->GetNeighbors();
		ASSERT_EQ(6, (int)neighbors.size());

		for(auto& neighbor : neighbors)
		{
			auto& np = neighbor->GetPosition();
			ASSERT_TRUE(
			        is_close(np, n1, 1e-11) ||
			        is_close(np, n2, 1e-11) ||
			        is_close(np, n3, 1e-11) ||
			        is_close(np, n4, 1e-11) ||
			        is_close(np, n5, 1e-11) ||
			        is_close(np, n6, 1e-11)
			        );
		}
	}

	// Update list again to make sure lists are being properly cleared.
	world.UpdateNeighborList();
	for(int i = 0; i < world.GetParticleCount(); ++i)
	{
		auto* particle = world.SelectParticle(i);
		const Position& coords = particle->GetPosition();

		// Check neighbors
		Position n1 =
		{(coords[0] == n) ? 1.0 : (double) coords[0] + 1.0, (double) coords[1], (double) coords[2]};
		Position n2 =
		{(coords[0] == 1) ? (double) n : coords[0] - 1.0, (double) coords[1], (double) coords[2]};
		Position n3 =
		{(double) coords[0], (coords[1] == n) ? 1.0 : (double) coords[1] + 1.0, (double) coords[2]};
		Position n4 =
		{(double) coords[0], (coords[1] == 1) ? (double) n : coords[1] - 1.0, (double) coords[2]};
		Position n5 =
		{(double) coords[0], (double) coords[1], (coords[2] == n) ? 1.0 : (double) coords[2] + 1.0};
		Position n6 =
		{(double) coords[0], (double) coords[1], (coords[2] == 1) ? (double) n : coords[2] - 1.0};

		auto& neighbors = particle->GetNeighbors();
		ASSERT_EQ(6, (int)neighbors.size());

		for(auto& neighbor : neighbors)
		{
			auto& np = neighbor->GetPosition();
			ASSERT_TRUE(
			        is_close(np, n1, 1e-11) ||
			        is_close(np, n2, 1e-11) ||
			        is_close(np, n3, 1e-11) ||
			        is_close(np, n4, 1e-11) ||
			        is_close(np, n5, 1e-11) ||
			        is_close(np, n6, 1e-11)
			        );
		}
	}
	
	// Check the composition of the world to make sure it's correct.
	auto& composition = world.GetComposition();

	ASSERT_EQ(9000, composition[site1.GetSpeciesID()]);
	ASSERT_EQ(9000, composition[site2.GetSpeciesID()]);
	ASSERT_EQ(9000, composition[site3.GetSpeciesID()]);

	// Test changing species, adding particle, removing particle. 
	Particle* pr = world.DrawRandomParticle();
	world.RemoveParticle(pr);
	ASSERT_EQ(8999, composition[pr->GetSpeciesID()]);

	world.AddParticle(pr);
	ASSERT_EQ(9000, composition[pr->GetSpeciesID()]);

	int previd = pr->GetSpeciesID();
	int newid = (previd == 0) ? 1 : 0;
	pr->SetSpeciesID(newid);
	ASSERT_EQ(9001, composition[newid]);
	ASSERT_EQ(8999, composition[previd]);
}

TEST(SimpleWorld, MoveParticleSemantics)
{
	int n = 30;
	World world(n, n, n, 1.0, 1.0);

	ASSERT_EQ(0, world.GetParticleCount());
	world.AddParticle(new Particle({0,0,0}, {1,0,0}, "E1"));
	ASSERT_EQ(1, world.GetParticleCount());
	ASSERT_EQ(1, world.GetPrimitiveCount());
	auto * p = world.SelectParticle(0);
	Director d {1, 0, 0};
	ASSERT_TRUE(is_close(d, p->GetDirector(), 1e-9));
}

TEST(SimpleWorld, VolumeScaling)
{
	int n = 30;
	World world(n, n, n, 1.0, 1.0);
	Particle site1({0, 0, 0}, {1, 0, 0}, "E1");
	world.PackWorld({&site1}, {1.0}, 500, 1.0);
	ASSERT_EQ(500, world.GetParticleCount());
	ASSERT_EQ(500, world.GetPrimitiveCount());

	// Get random coordinate and check it afterwards.
	auto box = world.GetHMatrix();
	Particle* p = world.DrawRandomParticle();
	Position newpos = 2.0*p->GetPosition(); // we will scale by 2
	world.SetVolume(8*world.GetVolume(), true);
	ASSERT_TRUE(is_close(newpos, p->GetPosition(),1e-11));
}