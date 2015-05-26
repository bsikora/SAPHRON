#include "SimBuilder.h"
#include "ForceFields/LebwohlLasherFF.h"
#include "Connectivities/P2SAConnectivity.h"
#include "Moves/FlipSpinMove.h"
#include "Moves/IdentityChangeMove.h"
#include "Moves/SpeciesSwapMove.h"
#include "Moves/SphereUnitVectorMove.h"
#include "CSVObserver.h"
#include "ConsoleObserver.h"

namespace SAPHRON
{
	bool SimBuilder::CheckType(std::string type, std::vector<std::string> types)
	{
		auto it = std::find(types.begin(), types.end(), type);
		if(it == types.end())
		{
			_emsgs.push_back("The specified type, " + type + ", is invalid.");

			std::ostringstream s;
			std::copy(types.begin(), types.end(), 
					  std::ostream_iterator<std::string>(s," "));
			_emsgs.push_back("  Valid entries are: " + s.str());

			return false;
		}

		return true;
	}

	bool SimBuilder::ValidateWorld(Json::Value world)
	{
		bool err = false;
		// World tag exists.
		if(!world)
		{
			_emsgs.push_back("No world has been specified.");
			return false;
		}

		// World type specified.
		std::string type = [&]() -> std::string {
			try{
				return world.get("type", "").asString();
			} catch(std::exception& e) {
				_emsgs.push_back("Type unserialization error: " + std::string(e.what()));
				err = true;
				return "";
			}
		}();

		if(type.length() == 0)
		{
			_emsgs.push_back("No world type has been specified.");
			err = true;
		}

		// Validate that the world type is a valid entry.
		std::vector<std::string> types =  {"Simple"};
		if(!CheckType(type, types))
			err = true;

		// Assign world type.
		_worldprops.type = type;

		// Simple world type.
		auto wsize = world["size"];
		if(type == types[0])
		{
			if(!wsize.isArray() || wsize.size() != 3)
			{
				_emsgs.push_back("The world size must be a 1x3 array.");
				err = true;
			}

			double xlength = [&]() -> double {
				try{
					return wsize[0].asDouble();
				} catch(std::exception& e)
				{
					_emsgs.push_back("X-length unserialization error : " + std::string(e.what()));
					err = true;
					return 0;
				}
			}();

			double ylength = [&]() -> double {
				try{
					return wsize[1].asDouble();
				} catch(std::exception& e)
				{
					_emsgs.push_back("Y-length unserialization error : " + std::string(e.what()));
					err = true;
					return 0;
				}
			}();

			double zlength = [&]() -> double {
				try{
					return wsize[2].asDouble();
				} catch(std::exception& e)
				{
					_emsgs.push_back("Z-length unserialization error : " + std::string(e.what()));
					err = true;
					return 0;
				}
			}();

			if(xlength <= 0 || 
			   ylength <= 0 || 
			   zlength <= 0)
			{
				_emsgs.push_back("All world size elements must be greater than zero.");
				err = true;	
			}

			// Assign world size. 
			_worldprops.xlength = xlength;
			_worldprops.ylength = ylength;
			_worldprops.zlength = zlength;
		}

		double rcutoff = [&]()-> double { 
			try {	
				return world.get("cutoff_radius", 0).asDouble(); 
			} catch(std::exception& e) {
				_emsgs.push_back("Cutoff radius unserialization error : " + std::string(e.what()));
				err = true;
				return 0;
			}
		}();
		
		if(!rcutoff)
		{
			_emsgs.push_back("A positive cutoff radius must be specified.");
			err = true;
		}
		if(rcutoff > wsize[0].asDouble()/2.0 || 
		   rcutoff > wsize[1].asDouble()/2.0 || 
		   rcutoff > wsize[2].asDouble()/2.0)
		{
			_emsgs.push_back("The cutoff radius cannot exceed half the shortest world size vector.");
			err = true;
		}

		// Assign cutoff radius. 
		_worldprops.rcutoff = rcutoff;

		int seed = [&]()-> int { 
			try {	
				return world.get("seed", 0).asInt(); 
			} catch(std::exception& e) {
				_emsgs.push_back("Seed unserialization error : " + std::string(e.what()));
				err = true;
				return 0;
			}
		}();

		if(!seed)
		{
			_nmsgs.push_back("No seed provided. Generating a random number.");
			_worldprops.seed = rand();
		}
		else if(seed <= 0)
		{
			_emsgs.push_back("Invalid seed specified. Seed cannot be less than or equal to zero.");
			err = true;
		}
		else 
			_worldprops.seed = seed;

		return !err;
	}

	bool SimBuilder::ValidateParticles(Json::Value components, Json::Value particles)
	{
		bool err = false;
		assert(_errors == false);

		// Necessary tags exist.
		if(!components)
		{
			_emsgs.push_back("No components have been specified.");
			return false;
		}

		if(!particles)
		{
			_emsgs.push_back("No particles have been specified.");
			return false;
		}

		// Build blueprint of "components" section.
		ConstructBlueprint(components, "");

		// Parse particles.
		for (int i = 0; i < (int)particles.size(); ++i)
		{
			std::string istr = std::to_string(i);

			std::string species = particles[i].getMemberNames()[0];
			auto particle = particles[i][species];

			// Make sure the species exists as a component.
			// TODO: check children.
			if(_blueprint.find(species) == _blueprint.end())
			{
				_emsgs.push_back("Particle " + istr + 
					" species, " + species + ", has not been delcared as a component.");
				return false;
			}

			// Get blueprint.
			auto blueprint = _blueprint[species];

			// Validate parent. 
			auto parent = [&]() -> std::string {
				try {
					return particle.get("parent", "").asString();
				} catch(std::exception& e) {
					err = true;
					_emsgs.push_back("Particle " + istr + 
					" parent unserialization error : " + std::string(e.what()));
					return "";
				}
			}();

			if(parent != blueprint.parent)
			{
				std::cout << "Parent: " << parent << " " << blueprint.parent << std::endl;
				_emsgs.push_back("Particle " + istr + " parent must match component definition.");
				err = true;
			}

			// Validate residue.
			auto residue = particle.get("residue", "").asString();

			// Make sure molecule is not specified. Only primitives/atoms. 
			if((int)blueprint.children.size() != 0)
			{
				_emsgs.push_back("Particle " + istr + " cannot have children. Check definition.");
				err = true;
			}

			// Validate position.
			if(!particle["position"].isArray() || particle["position"].size() != 3)
			{
				_emsgs.push_back("Particle " + istr + " position must be a 1x3 array.");
				err = true;
			}

			Position position = [&]() -> Position {
				try{
					double x = particle["position"][0].asDouble();
					double y = particle["position"][1].asDouble();
					double z = particle["position"][2].asDouble();
					return Position{x,y,z};

				} catch(std::exception& e)
				{
					_emsgs.push_back("Particle "  + 
						istr + " position unserialization error : " + 
						std::string(e.what()));
					err = true;
					return Position{NAN, NAN, NAN};
				}
			}();

			if(position.x == NAN || position.y == NAN || position.z == NAN)
			{
				_emsgs.push_back("Particle positions must be valid doubles.");
				err = true;	
			}

			if(position.x < 0 || 
			   position.y < 0 || 
			   position.z < 0 || 
			   position.x > _worldprops.xlength || 
			   position.y > _worldprops.ylength || 
			   position.z > _worldprops.zlength)
			{
				_emsgs.push_back("Particle " + istr + " positions must be between 0 and world size.");
				err = true;	
			}

			// Validate director.
			Director director({0.0, 0.0, 0.0});
			if(!particle["director"])
			{
			} 
			else 
			{
				if(!particle["director"].isArray() || particle["director"].size() != 3)
				{
					_emsgs.push_back("Particle " + istr + " director must be a 1x3 array.");
					err = true;
				}
				else 
				{
					director = [&]() -> Director {
						try{
							double x = particle["director"][0].asDouble();
							double y = particle["director"][1].asDouble();
							double z = particle["director"][2].asDouble();
							return Director{x,y,z};

						} catch(std::exception& e)
						{
							_emsgs.push_back("Particle "  + 
								istr + " director unserialization error : " + 
								std::string(e.what()));
							err = true;
							return Director{NAN, NAN, NAN};
						}
					}();
				}
			}

			// Check norm.
			double norm = sqrt(director[0]*director[0] + 
							   director[1]*director[1] + 
							   director[2]*director[2]);
			if(std::abs(norm - 1.0) > 1e-7 && norm != 0)
			{
				_nmsgs.push_back("Particle " + istr + " norm is not equal to 1. Normalizing vector.");
				director[0] /= norm;
				director[1] /= norm;
				director[2] /= norm;
			}

			// Push back struct 
			_particles.push_back({position, director, species, parent, residue, 0, {}});
		}

		return !err;
	}

	bool SimBuilder::ValidateForceFields(Json::Value forcefields)
	{
		bool err = false;
		assert(_errors == false);

		// Necessary tags exist.
		if(!forcefields)
		{
			_emsgs.push_back("No forcefields have been specified.");
			return false;
		}

		// Validate type.
		if(!forcefields.isArray())
		{
			_emsgs.push_back("Forcefield specification must be an array.");
			return false;
		}

		// Valid forcefields  . 
		std::vector<std::string> types = {"LebwohlLasher"};


		for (int i = 0; i < (int)forcefields.size(); ++i)
		{
			
			// forcefield type specified.
			std::string type = [&]() -> std::string {
				try{
					return forcefields[i].getMemberNames()[0];
				} catch(std::exception& e) {
					_emsgs.push_back("Type unserialization error: " + std::string(e.what()));
					err = true;
					return "";
				}
			}();

			auto forcefield = forcefields[i][type];

			if(type.length() == 0)
			{
				_emsgs.push_back("A forcefield has been incorrectly specified.");
				err = true;
			}
			else
			{

				if(!CheckType(type, types))
					err = true;
				else
				{
					ForceFieldProps ff;
					// Lebwohl Lasher
					if(type == types[0])
					{
						if(!forcefield["epsilon"] || !forcefield["gamma"])
						{
							_emsgs.push_back("Both epsilon and gamma must be specified for the " + type + " forcefield.");
							err = true;
						}

						double epsilon = [&]()-> double { 
							try {	
								return forcefield.get("epsilon", 0).asDouble(); 
							} catch(std::exception& e) {
								_emsgs.push_back("Epsilon unserialization error : " + std::string(e.what()));
								err = true;
								return 0;
							}
						}();	

						double gamma = [&]()-> double { 
							try {	
								return forcefield.get("gamma", 0).asDouble(); 
							} catch(std::exception& e) {
								_emsgs.push_back("Epsilon unserialization error : " + std::string(e.what()));
								err = true;
								return 0;
							}
						}();

						ff.type = type;
						ff.parameters.push_back(epsilon);
						ff.parameters.push_back(gamma);
					}

					// Validate species 
					auto species = forcefield["species"];
					if(!species || !species.isArray() || species.size() != 2)
					{
						_emsgs.push_back("Forcefield " + type  + " species is required and must be a 1x2 array.");
						err = true;
					}

					std::string species1 = [&]() -> std::string {
						try {
							return species[0].asString();
						} catch(std::exception& e) {
							err = true;
							_emsgs.push_back("Forcefield " + type + 
							" species unserialization error : " + std::string(e.what()));
							return "";
						}
					}();

					std::string species2 = [&]() -> std::string {
						try {
							return species[1].asString();
						} catch(std::exception& e) {
							err = true;
							_emsgs.push_back("Forcefield " + type + 
							" species unserialization error : " + std::string(e.what()));
							return "";
						}
					}();

					// Make sure species exist in our blueprint. 
					if(_blueprint.find(species1) == _blueprint.end())
					{
						_emsgs.push_back("Forcefield " + type + 
							" is defined for an undefined species " + species1 + ".");
						err = true;
					}

					if(_blueprint.find(species2) == _blueprint.end())
					{
						_emsgs.push_back("Forcefield " + type + 
							" is defined for an undefined species " + species2 + ".");
						err = true;
					}

					ff.species1 = species1;
					ff.species2 = species2;

					_forcefields.push_back(ff);
				}
			}
		}

		return !err;
	}

	bool SimBuilder::ValidateConnectivities(Json::Value connectivities)
	{
		bool err = false;
		assert(_errors == false);

		// Connectivities are optional.
		if(!connectivities)
		{
			return true; 
		}

		// Validate type.
		if(!connectivities.isArray())
		{
			_emsgs.push_back("Connectivity specification must be an array.");
			return false;
		}

		// Valid connectivities. 
		std::vector<std::string> types = {"P2SA"};

		for (int i = 0; i < (int)connectivities.size(); ++i)
		{
			
			// connectivity type specified.
			std::string type = [&]() -> std::string {
				try{
					return connectivities[i].getMemberNames()[0];
				} catch(std::exception& e) {
					_emsgs.push_back("Type unserialization error: " + std::string(e.what()));
					err = true;
					return "";
				}
			}();

			auto connectivity = connectivities[i][type];

			if(type.length() == 0)
			{
				_emsgs.push_back("A connectivity has been incorrectly specified.");
				err = true;
			}
			else
			{
				if(!CheckType(type, types))
					err = true;
				else
				{
					ConnectivityProps cprops;

					// P2SA
					if(type == types[0])
					{
						if(!connectivity.isMember("coefficient") || 
						   !connectivity.isMember("director"))
						{
							_emsgs.push_back(
								"Both coefficient and director must be specified for the " + 
								type + " connectivity."
							);
							err = true;
						}

						double coefficient = [&]()-> double { 
							try {	
								return connectivity.get("coefficient", 0).asDouble(); 
							} catch(std::exception& e) {
								_emsgs.push_back("Coefficient unserialization error : " + std::string(e.what()));
								err = true;
								return 0;
							}
						}();

						Director director({0.0, 0.0, 0.0});
						if(!connectivity["director"].isArray() || connectivity["director"].size() != 3)
						{
							_emsgs.push_back(type + " director must be a 1x3 array.");
							err = true;
						}	
						else
						{
							director = [&]() -> Director {
								try{
									double x = connectivity["director"][0].asDouble();
									double y = connectivity["director"][1].asDouble();
									double z = connectivity["director"][2].asDouble();
									return Director{x,y,z};

								} catch(std::exception& e)
								{
									_emsgs.push_back(type + " director unserialization error : " + 
										std::string(e.what()));
									err = true;
									return Director{NAN, NAN, NAN};
								}
							}();
						}

						// Check norm.
						double norm = sqrt(director[0]*director[0] + 
										   director[1]*director[1] + 
										   director[2]*director[2]);
						if(std::abs(norm - 1.0) > 1e-7 && norm != 0)
						{
							_nmsgs.push_back(type + " norm is not equal to 1. Normalizing vector.");
							director[0] /= norm;
							director[1] /= norm;
							director[2] /= norm;
						}

						cprops.parameters.push_back(coefficient);
						cprops.vparameters.push_back(director);
					} // end P2SA

					// General selector parsing for particles.
					auto particles = connectivity["particles"];
					if(!particles.isArray())
					{
						_emsgs.push_back(type + " selector must be an array.");
						err = true;
					}
					else if(particles.size() < 1)
					{
						_emsgs.push_back(type + " selector must contain at least 1 element.");
						err = true;
					}

					for(auto& selector : particles)
					{
						// Individual particle.
						if(selector.isInt())
						{
							int iselector = selector.asInt();
							if(iselector >= (int)_particles.size())
							{
								_emsgs.push_back(type +  
									" selector \"index" + std::to_string(iselector) + 
									"\" is invalid.");
								err = true;
							}
							else 
							{
								// Look up particle to check for duplicates.
								if(LookupIndexInConnectivity(iselector, cprops) || 
								   LookupParticleInConnectivity(_particles[i],cprops))
								{
									_emsgs.push_back(type + " duplicated for " + 
										"selector \"index " + std::to_string(iselector) + "\".");
									err = true;
								}
								else
									cprops.piselector.push_back(iselector);
							}				
						}
						// Array of particles.
						else if(selector.isArray())
						{
							int lower = [&]()-> int { 
								try {	
									return selector[0].asInt(); 
								} catch(std::exception& e) {
									_emsgs.push_back(type + 
										"selector unserialization error : " + 
										std::string(e.what()));
									err = true;
									return 0;
								}
							}();

							int upper = [&]()-> int { 
								try {	
									return selector[1].asInt(); 
								} catch(std::exception& e) {
									_emsgs.push_back(type + 
										"selector unserialization error : " + 
										std::string(e.what()));
									err = true;
									return 0;
								}
							}();

							for(int j = lower; j <= upper; ++j)
							{
								int iselector = j;
								if(iselector >= (int)_particles.size())
								{
									_emsgs.push_back(type +  
										" selector \"index " + std::to_string(iselector) + 
										"\" is invalid.");
									err = true;
								}
								else 
								{
									// Look up particle to check for duplicates.
									if(LookupIndexInConnectivity(iselector, cprops) || 
									   LookupParticleInConnectivity(_particles[i],cprops))
									{
										_emsgs.push_back(type + " duplicated for " + 
											"selector \"index " + std::to_string(iselector) + "\".");
										err = true;
									}
									else
										cprops.piselector.push_back(iselector);
								}			
							}

						}
						// Selector string. 
						else if(selector.isString())
						{
							std::string iselector = selector.asString();
							if(LookupStringInConnectivity(iselector, cprops))
									{
										_emsgs.push_back(type + " duplicated for " + 
											"selector " + iselector + ".");
										err = true;
									}
									else
										cprops.psselector.push_back(iselector);
						}
						else
						{
							_emsgs.push_back(type + " selector is invalid.");
							err = true;
						}
					} // End selector loop.
					cprops.type = type;
					_connectivities.push_back(cprops);
				}
			}
		}

		return !err;
	}

	bool SimBuilder::ValidateMoves(Json::Value moves)
	{
		assert(_errors == false);

		bool err = false;

		// Moves tag exists.
		if(!moves)
		{
			_emsgs.push_back("No moves have been specified.");
			return false;
		}

		// Move type specified.
		auto members = moves.getMemberNames();

		int seed = 0;


		// Valid move types.
		std::vector<std::string> types =  {
			"FlipSpin", "IdentityChange", "SpeciesSwap", 
			"SphereUnitVector"
		};

		// Go through members.
		for(auto& member : members)
		{
			// Main seed.
			if(member == "seed")
			{
				seed = [&]()-> int { 
					try {	
						return moves[member].asInt(); 
					} catch(std::exception& e) {
						_emsgs.push_back("Seed unserialization error : " + std::string(e.what()));
						err = true;
						return 0;
					}
				}();
			}
			else if(!CheckType(member, types))
				err = true;		
			else
			{
				MoveProps move;

				// Flip spin.
				if(member == types[0])
					move.type = types[0];
				// Identity change.
				else if(member == types[1])
					move.type = types[1];
				// Species swap.
				else if(member == types[2])
					move.type = types[2];
				else if(member == types[3])
					move.type = types[3];
				else // This is technically unneccesary because we check above. But good
					 // for debugging purposes.
				{
					_emsgs.push_back("The type of move specified is invalid.");

					std::ostringstream s;
					std::copy(types.begin(), types.end(), 
							  std::ostream_iterator<std::string>(s," "));
					_emsgs.push_back("    Valid entries are: " + s.str());

					err = true;
				}

				// Process seed for all types even if they don't need it. Easier this way.
				int moveseed = 0;
				if(!moves[member].isMember("seed"))
				{
					_nmsgs.push_back("No seed provided for " + member + 
						" move.\n     Generating a random number.");
					moveseed = rand();
				}
				else 
				{
					moveseed = [&]()-> int { 
						try {	
							return moves[member]["seed"].asInt(); 
						} catch(std::exception& e) {
							_emsgs.push_back("Seed unserialization error : " + std::string(e.what()));
							err = true;
							return 0;
						}
					}();
				}

				if(moveseed <= 0)
				{
					_emsgs.push_back("Invalid " + member + 
						" seed specified.\n     Seed cannot be less than or equal to zero.");
					err = true;
				}
		
				move.seed = moveseed;

				_moves.push_back(move);
			}
		}

		if(!seed)
		{
			_nmsgs.push_back("No seed provided. Generating a random number.");
			seed = rand();
		}
		else if(seed <= 0)
		{
			_emsgs.push_back("Invalid seed specified. Seed cannot be less than or equal to zero.");
			err = true;
		}
		else 
			_moveseed = seed;

		return !err;
	}

	bool SimBuilder::ValidateObservers(Json::Value observers)
	{
		assert(_errors == false);

		bool err = false; 

		if(!observers)
		{
			_emsgs.push_back("At least one observer must be specified");
			return false;
		}

		// World type specified.
		auto members = observers.getMemberNames();

		// Valid observer types.
		std::vector<std::string> types =  {
			"Console", "CSV"
		};

		// Go through observers.
		for(auto& member: members)
		{
			if(!CheckType(member, types))
				err = true;
			else
			{
				ObserverProps obs;

				// Console observer.
				if(member == types[0])
				{
					obs.type = types[0];	
				} 
				// CSV observer.
				else if(member == types[1])
				{
					obs.type = types[1];

					obs.prefix = [&]() -> std::string {
						try{
							return observers[member].get("file_prefix", "").asString();
						} catch(std::exception& e) {
							_emsgs.push_back("File prefix unserialization error: " + std::string(e.what()));
							err = true;
							return "";
						}
					}();

					if((int)obs.prefix.size() == 0)
					{
						_emsgs.push_back(member + " observer file prefix must be specified.");
						err = true;
					}
				}


				// Get frequency spec.
				obs.frequency = [&]()-> int { 
					try {	
						return observers[member]["frequency"].asInt(); 
					} catch(std::exception& e) {
						_emsgs.push_back("Frequency unserialization error : " + std::string(e.what()));
						err = true;
						return 0;
					}
				}();

				if(obs.frequency == 0)
				{
					_emsgs.push_back(member + " frequency must be a positive integer.");
					err = true;
				}

				if(!observers[member].isMember("flags"))
				{
					_emsgs.push_back(member + " flags must be specified.");
					err = true;
				}
				else
				{
					SimFlags sflags;

					auto flagtree = observers[member]["flags"];
					sflags.identifier = ProcessFlag(flagtree, "identifier", member, err);
					sflags.iterations = ProcessFlag(flagtree, "iterations", member, err);
					sflags.energy = ProcessFlag(flagtree, "energy", member, err);
					sflags.temperature = ProcessFlag(flagtree, "temperature", member, err);
					sflags.pressure = ProcessFlag(flagtree, "pressure", member, err);
					sflags.composition = ProcessFlag(flagtree, "composition", member, err);
					sflags.acceptance = ProcessFlag(flagtree, "acceptance", member, err);
					sflags.dos_walker = ProcessFlag(flagtree, "dos_walker", member, err);
					sflags.dos_scale_factor = ProcessFlag(flagtree, "dos_scale_factor", member, err);
					sflags.dos_flatness = ProcessFlag(flagtree, "dos_flatness", member, err);
					sflags.dos_interval = ProcessFlag(flagtree, "dos_interval", member, err);
					sflags.dos_bin_count = ProcessFlag(flagtree, "dos_bin_count", member, err);
					sflags.dos_values = ProcessFlag(flagtree, "dos_values", member, err);
					sflags.particle_global_id = ProcessFlag(flagtree, "particle_global_id", member, err);
					sflags.particle_position = ProcessFlag(flagtree, "particle_position", member, err);
					sflags.particle_director = ProcessFlag(flagtree, "particle_director", member, err);
					sflags.particle_species = ProcessFlag(flagtree, "particle_species", member, err);
					sflags.particle_species_id = ProcessFlag(flagtree, "particle_species_id", member, err);
					sflags.particle_neighbors = ProcessFlag(flagtree, "particle_neighbors", member, err);
				}

				_observers.push_back(obs);
			}
		}

		return !err;
	}

	bool SimBuilder::ValidateEnsemble(Json::Value ensemble)
	{
		bool err = false;

		if(!ensemble)
		{
			_emsgs.push_back("No ensemble has been specified");
			return false;
		}

		// Ensemble type specified.
		std::string type = [&]() -> std::string {
			try{
				return ensemble.get("type", "").asString();
			} catch(std::exception& e) {
				_emsgs.push_back("Type unserialization error: " + std::string(e.what()));
				err = true;
				return "";
			}
		}();

		if(type.length() == 0)
		{
			_emsgs.push_back("No ensemble type has been specified.");
			err = true;
		}

		std::vector<std::string> types =  {"NVT"};
		if(!CheckType(type, types))
			err = true;

		_ensemble.type = type;

		// Sweeps
		int sweeps = [&]()-> int { 
			try {	
				return ensemble.get("sweeps", 0).asInt(); 
			} catch(std::exception& e) {
				_emsgs.push_back("Sweeps unserialization error : " + std::string(e.what()));
				err = true;
				return 0;
			}
		}();

		if(sweeps <= 0)
		{
			_emsgs.push_back("Invalid sweeps specified. Sweeps cannot be less than or equal to zero.");
			err = true;
		}
		else 
			_ensemble.sweeps = sweeps;

		// Seed
		int seed = [&]()-> int { 
			try {	
				return ensemble.get("seed", 0).asInt(); 
			} catch(std::exception& e) {
				_emsgs.push_back("Seed unserialization error : " + std::string(e.what()));
				err = true;
				return 0;
			}
		}();

		if(!seed)
		{
			_nmsgs.push_back("No seed provided. Generating a random number.");
			_ensemble.seed = rand();
		}
		else if(seed <= 0)
		{
			_emsgs.push_back("Invalid seed specified. Seed cannot be less than or equal to zero.");
			err = true;
		}
		else 
			_ensemble.seed = seed;

		return !err;
	}

	int SimBuilder::ProcessFlag(Json::Value& flagtree, std::string flag, std::string observer, bool& err)
	{
		return [&]()-> int { 
			try {	
				return flagtree.get(flag, 0).asInt() != 0 ? 1 : 0; 
			} catch(std::exception& e) {
				_emsgs.push_back(
					observer + " identifier flag " + flag + " unserialization error : " 
					+ std::string(e.what()));
				err = true;
				return 0;
			}
		}();
	}

	bool SimBuilder::LookupParticleInConnectivity(ParticleProps& particle, ConnectivityProps& connectivity)
	{
		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), particle.species) != 
			connectivity.psselector.end())
			return true;

		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), particle.residue) != 
			connectivity.psselector.end())
			return true;

		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), particle.parent) != 
			connectivity.psselector.end())
			return true;

		return false;
	}

	bool SimBuilder::LookupStringInConnectivity(std::string keyword, ConnectivityProps& connectivity)
	{
		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), keyword) != 
			connectivity.psselector.end())
			return true;

		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), keyword) != 
			connectivity.psselector.end())
			return true;

		if(std::find(connectivity.psselector.begin(), connectivity.psselector.end(), keyword) != 
			connectivity.psselector.end())
			return true;

		for(auto &index : connectivity.piselector)
		{
			if(_particles[index].species == keyword)
				return true;
			if(_particles[index].residue == keyword)
				return true;
			if(_particles[index].parent == keyword)
				return true;
		}

		return false;
	}


	bool SimBuilder::LookupIndexInConnectivity(int index, ConnectivityProps& connectivity)
	{
		if(std::find(connectivity.piselector.begin(),connectivity.piselector.end(), index) != 
			connectivity.piselector.end())
			return true;

		return false;
	}

	bool SimBuilder::ConstructBlueprint(Json::Value components, std::string parent)
	{
		bool err = false;

		auto species = components.getMemberNames();
		for(int i = 0; i < (int)components.size(); ++i)
		{
			auto component = components[species[i]];
			// Species is new, add to blueprint.
			if(_blueprint.find(species[i]) == _blueprint.end())
			{
				ParticleProps blueprint;
				blueprint.species = species[i];
				blueprint.parent = parent;

				blueprint.index = [&]() -> int {
					try {
						return component.get("index", 0).asInt();
					} catch(std::exception& e) {
						err = true;
						_emsgs.push_back("Component" + species[i] + "index unserialization error : " + 
							std::string(e.what()));
						return 0;
					}
				}();

				// Enumerate children.
				if(component.isMember("children"))
				{
					auto children = component["children"];
					if(!children.isArray())
					{
						_emsgs.push_back("Component " + species[i] + " children must be an array.");
						err = true;
					}
					else 
					{
						for(int j = 0; j < (int)children.size(); ++j)
						{							
							ConstructBlueprint(children[j], species[i]);
							blueprint.children.push_back(children[j].getMemberNames()[0]);
						}
					}
				}
				_blueprint[species[i]] = blueprint;
				_nmsgs.push_back("Created blueprint for " + species[i] + ".");
			}
		}

		return !err;
	}

	World* SimBuilder::BuildWorld()
	{
		assert(_errors == false);

		if(_worldprops.type == "Simple")
		{
			_nmsgs.push_back("Setting size to [" +  
				std::to_string(_worldprops.xlength) + ", " + 
				std::to_string(_worldprops.ylength) + ", " + 
				std::to_string(_worldprops.zlength) + "] \u212B.");
			_nmsgs.push_back("Setting cutoff radius to " + std::to_string(_worldprops.rcutoff) + " \u212B.");
			_nmsgs.push_back("Setting seed to " + std::to_string(_worldprops.seed) + ".");
			return new SimpleWorld(_worldprops.xlength, 
								   _worldprops.ylength, 
								   _worldprops.zlength, 
								   _worldprops.rcutoff,
								   _worldprops.seed);
		}

		return nullptr;
	}

	void SimBuilder::BuildParticles(World* world, std::vector<Connectivity*>& connectivities)
	{
		assert(_errors == false);
		

		// Initialize connectivities. 
		for(auto& connectivity : _connectivities)
		{
			if(connectivity.type == "P2SA")
			{
				connectivities.push_back(
					new P2SAConnectivity(connectivity.parameters[0],connectivity.vparameters[0])
				);
			}
		}

		assert(_connectivities.size() == connectivities.size());

		std::map<std::string, int> ccounts; // Connectivity counts.
		std::map<std::string, int> pcounts; // Particle counts.
		for (int i = 0; i < (int)_particles.size(); ++i)
		{
			auto particle = _particles[i];

			// Count.
			if(pcounts.find(particle.species) == pcounts.end())
				pcounts[particle.species] = 1;
			else
				++pcounts[particle.species];

			// Initialize particles.
			Particle* p = new Site(particle.position, particle.director, particle.species);
			_ppointers.push_back(p);
			world->AddParticle(p);

			// See if there is any associated connectivity.
			for(int j = 0; j < (int)connectivities.size(); ++j)
			{
				auto connectivity = _connectivities[j];
				if(LookupParticleInConnectivity(particle, connectivity) || 
				   LookupIndexInConnectivity(i, connectivity))
				{
					p->AddConnectivity(connectivities[j]);
					if(ccounts.find(connectivity.type) == ccounts.end())
						ccounts[connectivity.type] = 1;
					else
						++ccounts[connectivity.type];
				}
			}
		}

		for(auto& count : pcounts)
			_nmsgs.push_back("Initialized " + 
				std::to_string(count.second) + " particle(s) of type " + count.first + ".");

		for(auto& count : ccounts)
			_nmsgs.push_back("Added " + std::to_string(count.second) + 
				" particle(s) to " + count.first + " connectivity.");

		assert(_particles.size() == _ppointers.size());
	}

	void SimBuilder::BuildForceFields(std::vector<ForceField*>& forcefields, ForceFieldManager& ffm)
	{
		assert(_errors == false);

		for(auto& forcefield : _forcefields)
		{
			if(forcefield.type == "LebwohlLasher")
			{
				ForceField* ff = new LebwohlLasherFF(forcefield.parameters[0], 
													 forcefield.parameters[1]);
				
				ffm.AddForceField(forcefield.species1, forcefield.species2, *ff);
				forcefields.push_back(ff);
				_nmsgs.push_back("Initialized " + forcefield.type + " force field for [" + 
					forcefield.species1 + "," + forcefield.species2 + 
					"]\n     with epsilon = "  + std::to_string(forcefield.parameters[0]) + 
					" and gamma = " + std::to_string(forcefield.parameters[1]) + ".");
			}
		}
	}

	void SimBuilder::BuildMoves(std::vector<Move*>& moves, MoveManager& mm)
	{
		assert(_errors == false);

		mm.SetSeed(_moveseed);
		_nmsgs.push_back("Set move manager seed to " + std::to_string(_moveseed) + ".");

		int pcount = 0;
		// Count primitive types (site).
		for(auto& particle : _blueprint)
			if((int)particle.second.children.size() == 0)
				pcount++;

		for(auto& move : _moves)
		{
			Move* moveptr = nullptr;
			if(move.type == "Flipspin")
			{
				moveptr = new FlipSpinMove();
				_nmsgs.push_back("Initialized " + move.type + " move.");
			}
			else if(move.type == "IdentityChange")
			{
				moveptr = new IdentityChangeMove(pcount, move.seed);
				_nmsgs.push_back("Initialized " + move.type  + " move with nspecies = " + 
					std::to_string(pcount) + "\n     and seed = " + 
					std::to_string(move.seed) + ".");
			}
			else if(move.type == "SpeciesSwap")
			{
				moveptr = new SpeciesSwapMove();
				_nmsgs.push_back("Initialized " + move.type + " move.");
			}
			else if(move.type == "SphereUnitVector")
			{
				moveptr = new SphereUnitVectorMove(move.seed);
				_nmsgs.push_back("Initialized " + move.type +
				 " move with seed = " + std::to_string(move.seed) + ".");
			}

			if(moveptr != nullptr)
			{
				moves.push_back(moveptr);
				mm.PushMove(moveptr);
			}
		}
	}

	void SimBuilder::BuildObservers(std::vector<SimObserver*>& observers)
	{
		assert(_errors == false);

		for(auto& observer : _observers)
		{
			if(observer.type == "CSV")
			{
				observers.push_back(
					new CSVObserver(observer.prefix,
									observer.flags, 
									observer.frequency)
				);

				_nmsgs.push_back("Initilaized " + observer.type + 
					" observer with frequency = " + std::to_string(observer.frequency) + 
					"\n     and file prefix = " + observer.prefix + ".");

			}
			else if(observer.type == "Console")
			{
				observers.push_back(
					new ConsoleObserver(observer.flags,
										observer.frequency)
				);

				_nmsgs.push_back("Initilaized " + observer.type + 
					" observer with frequency = " + std::to_string(observer.frequency) + 
					".");
			}
		}

		assert(observers.size() == _observers.size());
	}
}