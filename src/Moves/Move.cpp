#include <random>
#include "Move.h"
#include "MoveManager.h"
#include "json/json.h"
#include "schema.h"
#include "../Validator/ObjectRequirement.h"
#include "../Validator/ArrayRequirement.h"
#include "../Simulation/SimException.h"
#include "../Worlds/WorldManager.h"
#include "FlipSpinMove.h"
#include "TranslateMove.h"
#include "TranslatePrimitiveMove.h"
#include "DirectorRotateMove.h"
#include "ParticleSwapMove.h"
#include "RandomIdentityMove.h"
#include "RotateMove.h"
#include "SpeciesSwapMove.h"
#include "VolumeSwapMove.h"
#include "VolumeScaleMove.h"
#include "InsertParticleMove.h"
#include "DeleteParticleMove.h"
#include "AnnealChargeMove.h"
#include "AcidTitrationMove.h"
#include "AcidReactionMove.h"
#include "WidomInsertionMove.h"
#include "RegionInsertParticleMove.h"
#include "ConfinementDeleteParticleMove.h"
#include "ConfinementAcidReactionMove.h"
#include "TubeConfinementInsertParticleMove.h"
#include "TubeConfinementDeleteParticleMove.h"
#include "TubeConfinementAcidReactionMove.h"

#include "MODInsertParticleMove.h"
#include "MODDeleteParticleMove.h"

#ifdef USING_LAMMPS
#include "MDMove.h"
#include "MDMoveSSages.h"
#include "MDMoveMultiCore.h"
#endif
using namespace Json;

namespace SAPHRON
{
	Move* Move::BuildMove(const Json::Value &json, SAPHRON::MoveManager *mm, WorldManager* wm)
	{
		return BuildMove(json, mm, wm, "#/moves");
	}

	Move* Move::BuildMove(const Value &json, 
						  MoveManager *mm, 
						  WorldManager* wm, 
						  const std::string& path)
	{
		ObjectRequirement validator;
		Value schema;
		Reader reader;

		Move* move = nullptr;

		// Random device for seed generation. 
		std::random_device rd;
		auto maxi = std::numeric_limits<int>::max();
		auto seed = json.get("seed", rd() % maxi).asUInt();

		// Get move type. 
		std::string type = json.get("type", "none").asString();

		if(type == "AcidReaction")
		{
			reader.parse(JsonSchema::AcidReactionMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> reactants;
			for(auto& s : json["swap"])
				reactants.push_back(s.asString());

			std::vector<std::string> products;
			for(auto& s : json["products"])
				products.push_back(s.asString());

			auto pKo = json.get("mu", 0.0).asDouble();
			auto scount = json["stash_count"].asInt();

			auto prefac = json.get("op_prefactor", true).asBool();

			auto* m = new AcidReactionMove(reactants,products,*wm,
			scount, pKo, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "ConfinementAcidReaction")
		{
			reader.parse(JsonSchema::ConfinementAcidReactionMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> reactants;
			for(auto& s : json["swap"])
				reactants.push_back(s.asString());

			std::vector<std::string> products;
			for(auto& s : json["products"])
				products.push_back(s.asString());

			auto pKo = json.get("mu", 0.0).asDouble();
			auto scount = json["stash_count"].asInt();

			auto prefac = json.get("op_prefactor", true).asBool();

			auto x_low = json.get("xlo", 0.0).asDouble();
			auto x_high = json.get("xhi", 0.0).asDouble();

			auto y_low = json.get("ylo", 0.0).asDouble();
			auto y_high = json.get("yhi", 0.0).asDouble();

			auto z_low = json.get("zlo", 0.0).asDouble();
			auto z_high = json.get("zhi", 0.0).asDouble();

			auto* m = new ConfinementAcidReactionMove(reactants,products,*wm,
			scount, pKo, x_low, y_low, z_low, x_high, y_high, z_high, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "TubeConfinementAcidReaction")
		{
			reader.parse(JsonSchema::TubeConfinementAcidReactionMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> reactants;
			for(auto& s : json["swap"])
				reactants.push_back(s.asString());

			std::vector<std::string> products;
			for(auto& s : json["products"])
				products.push_back(s.asString());

			auto pKo = json.get("mu", 0.0).asDouble();
			auto scount = json["stash_count"].asInt();

			auto prefac = json.get("op_prefactor", true).asBool();

			auto cylin_rad = json.get("cylrad", 0.0).asDouble();
			auto cylin_y = json.get("cylcen_y", 0.0).asDouble();
			auto cylin_z = json.get("cylcen_z", 0.0).asDouble();

			auto* m = new TubeConfinementAcidReactionMove(reactants,products,*wm,
			scount, pKo,cylin_rad, cylin_y, cylin_z, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "AcidTitrate")
		{
			reader.parse(JsonSchema::AcidTitrationMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto protoncharge = json.get("proton_charge", 1.0).asDouble();
			auto mu = json.get("mu", 0.0).asDouble();
			auto prefac = json.get("op_prefactor", true).asBool();

			auto* m = new AcidTitrationMove(species, protoncharge, mu, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "AnnealCharge")
		{
			reader.parse(JsonSchema::AnnealChargeMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			move = new AnnealChargeMove(species, seed);
		}
		else if(type == "DeleteParticle")
		{
			reader.parse(JsonSchema::DeleteParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_d = json.get("multi_delete", false).asBool();
			
			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto* m = new DeleteParticleMove(species, multi_d, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "MODDeleteParticle")
		{
			reader.parse(JsonSchema::MODDeleteParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_d = json.get("multi_delete", false).asBool();
			
			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			std::vector<std::string> ExtraCount;
			for(auto& s : json["extra_species"])
				ExtraCount.push_back(s.asString());

			auto* m = new MODDeleteParticleMove(species, ExtraCount, multi_d, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "ConfinementDeleteParticle")
		{
			reader.parse(JsonSchema::ConfinementDeleteParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_d = json.get("multi_delete", false).asBool();
			
			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto x_low = json.get("xlo", 0.0).asDouble();
			auto x_high = json.get("xhi", 0.0).asDouble();

			auto y_low = json.get("ylo", 0.0).asDouble();
			auto y_high = json.get("yhi", 0.0).asDouble();

			auto z_low = json.get("zlo", 0.0).asDouble();
			auto z_high = json.get("zhi", 0.0).asDouble();

			auto* m = new ConfinementDeleteParticleMove(species, multi_d, x_low, y_low, z_low, x_high, y_high, z_high, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "TubeConfinementDeleteParticle")
		{
			reader.parse(JsonSchema::TubeConfinementDeleteParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_d = json.get("multi_delete", false).asBool(); 

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto cylin_rad = json.get("cylrad", 0.0).asDouble();

			auto* m = new TubeConfinementDeleteParticleMove(species, multi_d, cylin_rad, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "DirectorRotate")
		{
			reader.parse(JsonSchema::DirectorRotateMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			move = new DirectorRotateMove(seed);
		}
		else if(type == "FlipSpin")
		{
			reader.parse(JsonSchema::FlipSpinMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			move = new FlipSpinMove(seed);
		}
		else if(type == "InsertParticle")
		{
			reader.parse(JsonSchema::InsertParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto scount = json["stash_count"].asInt();
			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_i = json.get("multi_insertion", false).asBool(); 

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto* m = new InsertParticleMove(species, *wm, scount, multi_i, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "MODInsertParticle")
		{
			reader.parse(JsonSchema::MODInsertParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto scount = json["stash_count"].asInt();
			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_i = json.get("multi_insertion", false).asBool(); 

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			std::vector<std::string> ExtraCount;
			for(auto& s : json["extra_species"])
				ExtraCount.push_back(s.asString());

			auto* m = new MODInsertParticleMove(species, ExtraCount, *wm, scount, multi_i, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "RegionInsertParticle")
		{
			reader.parse(JsonSchema::RegionInsertParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto scount = json["stash_count"].asInt();
			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_i = json.get("multi_insertion", false).asBool(); 

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto x_low = json.get("xlo", 0.0).asDouble();
			auto x_high = json.get("xhi", 0.0).asDouble();

			auto y_low = json.get("ylo", 0.0).asDouble();
			auto y_high = json.get("yhi", 0.0).asDouble();

			auto z_low = json.get("zlo", 0.0).asDouble();
			auto z_high = json.get("zhi", 0.0).asDouble();

			auto* m = new RegionInsertParticleMove(species, *wm, scount, multi_i, x_low, y_low, z_low, x_high, y_high, z_high, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "TubeConfinementInsertParticle")
		{
			reader.parse(JsonSchema::TubeConfinementInsertParticleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto scount = json["stash_count"].asInt();
			auto prefac = json.get("op_prefactor", true).asBool();
			auto multi_i = json.get("multi_insertion", false).asBool(); 

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			auto cylin_rad = json.get("cylrad", 0.0).asDouble();
			auto cylin_y = json.get("cylcen_y", 0.0).asDouble();
			auto cylin_z = json.get("cylcen_z", 0.0).asDouble();

			auto* m = new TubeConfinementInsertParticleMove(species, *wm, scount, multi_i, cylin_rad, cylin_y, cylin_z, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		#ifdef USING_LAMMPS
		else if(type == "MolecularDynamics")
		{
			reader.parse(JsonSchema::MDMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto datafile = json.get("data_file","none").asString();
			auto inputfile = json.get("input_file","none").asString();
			auto minimizefile = json.get("minimize_file","none").asString();
			
			std::vector<std::string> sidentities;
			std::vector<int> lidentities;

			auto& mapping = json["mapping_ids"];
			for(auto& map : mapping)
			{
				sidentities.push_back(map[0].asString());
				lidentities.push_back(map[1].asInt());
			}

			auto* m = new MDMove(datafile, inputfile, minimizefile, sidentities, lidentities, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		#endif
		else if(type == "MolecularDynamicsSSages")
		{
			reader.parse(JsonSchema::MDMoveSSages, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto datafile = json.get("data_file","none").asString();
			auto inputfile = json.get("input_file","none").asString();
			auto minimizefile = json.get("minimize_file","none").asString();
			auto ssagesfile = json.get("ssages_file","none").asString();
			auto numproc = json.get("num_proc",1).asInt();
			
			std::vector<std::string> sidentities;
			std::vector<int> lidentities;

			auto& mapping = json["mapping_ids"];
			for(auto& map : mapping)
			{
				sidentities.push_back(map[0].asString());
				lidentities.push_back(map[1].asInt());
			}

			auto* m = new MDMoveSSages(datafile, inputfile, minimizefile, ssagesfile, numproc,sidentities, lidentities, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "MolecularDynamicsMultiCore")
		{
			reader.parse(JsonSchema::MDMoveMultiCore, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto prefac = json.get("op_prefactor", true).asBool();
			auto datafile = json.get("data_file","none").asString();
			auto inputfile = json.get("input_file","none").asString();
			auto minimizefile = json.get("minimize_file","none").asString();
			auto numproc = json.get("num_proc",1).asInt();
			
			std::vector<std::string> sidentities;
			std::vector<int> lidentities;

			auto& mapping = json["mapping_ids"];
			for(auto& map : mapping)
			{
				sidentities.push_back(map[0].asString());
				lidentities.push_back(map[1].asInt());
			}

			auto* m = new MDMoveMultiCore(datafile, inputfile, minimizefile, numproc,sidentities, lidentities, seed);
			m->SetOrderParameterPrefactor(prefac);
			move = static_cast<Move*>(m);
		}
		else if(type == "ParticleSwap")
		{
			reader.parse(JsonSchema::ParticleSwapMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			move = new ParticleSwapMove(seed);
		}
		else if(type == "RandomIdentity")
		{
			reader.parse(JsonSchema::RandomIdentityMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> identities;
			for(auto& i : json["identities"])
				identities.push_back(i.asString());

			move = new RandomIdentityMove(identities, seed);
		}
		else if(type == "Rotate")
		{
			reader.parse(JsonSchema::RotateMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			double dmax = json["maxangle"].asDouble();

			move = new RotateMove(dmax, seed);
		}
		else if(type == "SpeciesSwap")
		{
			reader.parse(JsonSchema::SpeciesSwapMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			bool deepcopy = json.get("deep_copy", false).asBool();

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			move = new SpeciesSwapMove(species,deepcopy,seed);
		}
		else if(type == "Translate")
		{
			reader.parse(JsonSchema::TranslateMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());
		
			if(json["dx"].isObject())
			{
				std::map<std::string, double> dx; 
				for(auto& s : json["dx"].getMemberNames())
					dx[s] = json["dx"][s].asDouble();

				auto expl = json.get("explicit_draw", false).asBool();

				move = new TranslateMove(dx, expl, seed);
			}
			else
			{
				auto dx = json["dx"].asDouble();
				move = new TranslateMove(dx, seed);
			}
		}
		else if(type == "TranslatePrimitive")
		{
			reader.parse(JsonSchema::TranslatePrimitiveMove, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());
		
			if(json["dx"].isObject())
			{
				std::map<std::string, double> dx; 
				for(auto& s : json["dx"].getMemberNames())
					dx[s] = json["dx"][s].asDouble();

				auto expl = json.get("explicit_draw", false).asBool();

				move = new TranslatePrimitiveMove(dx, expl, seed);
			}
			else
			{
				auto dx = json["dx"].asDouble();
				move = new TranslatePrimitiveMove(dx, seed);
			}
		}
		else if(type == "VolumeScale")
		{
			reader.parse(JsonSchema::VolumeScaleMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			auto dv = json["dv"].asDouble();
			auto Pextern = json["Pextern"].asDouble();

			move = new VolumeScaleMove(Pextern, dv, seed);		
		}
		else if(type == "VolumeSwap")
		{
			reader.parse(JsonSchema::VolumeSwapMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			double dv = json["dv"].asDouble();

			move = new VolumeSwapMove(dv, seed);
		}
		else if(type == "WidomInsertion")
		{
			reader.parse(JsonSchema::WidomInsertionMove, schema);
			validator.Parse(schema, path);

			// Validate inputs. 
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			std::vector<std::string> species;
			for(auto& s : json["species"])
				species.push_back(s.asString());

			move = new WidomInsertionMove(species, *wm, seed);
		}	
		else
		{
			throw BuildException({path + ": Unknown move type specified."});
		}

		// Add to appropriate species pair.
		try{
				int weight = json.get("weight", 1).asUInt();
				mm->AddMove(move, weight);
			} catch(std::exception& e) {
				delete move;
				throw BuildException({
					e.what()
				});
		}

		return move;
	}

	void Move::BuildMoves(const Json::Value &json, 
						  SAPHRON::MoveManager *mm, 
						  WorldManager* wm, 
						  MoveList &mvlist)
	{
		ArrayRequirement validator;
		Value schema;
		Reader reader;

		reader.parse(JsonSchema::Moves, schema);
		validator.Parse(schema, "#/moves");

		// Validate high level schema.
		validator.Validate(json, "#/moves");
		if(validator.HasErrors())
			throw BuildException(validator.GetErrors());

		// Loop through moves.
		int i = 0;
		for(auto& m : json)
		{
			mvlist.push_back(BuildMove(m, mm, wm, "#/moves/" + std::to_string(i)));
			++i;
		}

	}
}