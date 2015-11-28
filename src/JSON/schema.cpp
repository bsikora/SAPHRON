#include "schema.h"

namespace SAPHRON
{
	//INSERT_DEF_HERE
	std::string SAPHRON::JsonSchema::ElasticCoeffOP = "{\"required\": [\"type\", \"mode\", \"xrange\", \"world\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"mode\": {\"enum\": [\"twist\"], \"type\": \"string\"}, \"world\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"ElasticCoeff\"], \"type\": \"string\"}, \"xrange\": {\"maxItems\": 2, \"items\": {\"type\": \"number\"}, \"type\": \"array\", \"minItems\": 2}}}";
	std::string SAPHRON::JsonSchema::WangLandauOP = "{\"required\": [\"type\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"type\": {\"enum\": [\"WangLandau\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::Histogram = "{\"required\": [\"min\", \"max\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"max\": {\"type\": \"number\"}, \"values\": {\"items\": {\"type\": \"number\"}, \"type\": \"array\"}, \"binwidth\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"min\": {\"type\": \"number\"}, \"counts\": {\"items\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": \"array\"}, \"bincount\": {\"minimum\": 1, \"type\": \"integer\"}}}";
	std::string SAPHRON::JsonSchema::DOSSimulation = "{\"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"convergence_factor\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"reset_freq\": {\"minimum\": 0, \"type\": \"integer\"}, \"target_flatness\": {\"exclusiveMaximum\": true, \"minimum\": 0, \"maximum\": 1, \"exclusiveMinimum\": true, \"type\": \"number\"}}}";
	std::string SAPHRON::JsonSchema::Simulation = "{\"required\": [\"simtype\", \"iterations\"], \"type\": \"object\", \"properties\": {\"units\": {\"enum\": [\"real\", \"reduced\"], \"type\": \"string\"}, \"mpi\": {\"minimum\": 1, \"type\": \"integer\"}, \"simtype\": {\"enum\": [\"standard\", \"DOS\"], \"type\": \"string\"}, \"iterations\": {\"minimum\": 1, \"type\": \"integer\"}}}";
	std::string SAPHRON::JsonSchema::ForceFields = "{\"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"bonded\": {\"type\": \"array\"}, \"nonbonded\": {\"type\": \"array\"}, \"electrostatic\": {\"type\": \"array\"}}}";
	std::string SAPHRON::JsonSchema::DebyeHuckelFF = "{\"required\": [\"type\", \"kappa\", \"species\", \"rcut\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"species\": {\"maxItems\": 2, \"additionalItems\": false, \"minItems\": 2, \"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"kappa\": {\"minimum\": 0, \"type\": \"number\"}, \"rcut\": {\"items\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"type\": \"array\", \"minItems\": 1}, \"type\": {\"enum\": [\"DebyeHuckel\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::DSFFF = "{\"required\": [\"type\", \"alpha\", \"species\", \"rcut\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"alpha\": {\"minimum\": 0, \"type\": \"number\"}, \"rcut\": {\"items\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"type\": \"array\", \"minItems\": 1}, \"species\": {\"maxItems\": 2, \"additionalItems\": false, \"minItems\": 2, \"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"type\": {\"enum\": [\"DSF\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::LebwholLasherFF = "{\"required\": [\"type\", \"epsilon\", \"gamma\", \"species\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"epsilon\": {\"type\": \"number\"}, \"gamma\": {\"type\": \"number\"}, \"species\": {\"maxItems\": 2, \"additionalItems\": false, \"minItems\": 2, \"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"type\": {\"enum\": [\"LebwohlLasher\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::LennardJonesFF = "{\"required\": [\"type\", \"sigma\", \"epsilon\", \"species\", \"rcut\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"sigma\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"epsilon\": {\"minimum\": 0, \"type\": \"number\"}, \"rcut\": {\"items\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"type\": \"array\", \"minItems\": 1}, \"species\": {\"maxItems\": 2, \"additionalItems\": false, \"minItems\": 2, \"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"type\": {\"enum\": [\"LennardJones\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::Components = "{\"components\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"required\": [\"count\"], \"type\": \"object\", \"properties\": {\"count\": {\"minimum\": 1, \"type\": \"integer\"}}}}, \"additionalProperties\": false, \"minProperties\": 1, \"type\": \"object\"}}";
	std::string SAPHRON::JsonSchema::Worlds = "{\"items\": {\"required\": [\"type\", \"dimensions\", \"nlist_cutoff\", \"skin_thickness\", \"components\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"components\": {\"components\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"required\": [\"count\"], \"type\": \"object\", \"properties\": {\"count\": {\"minimum\": 1, \"type\": \"integer\"}}}}, \"additionalProperties\": false, \"minProperties\": 1, \"type\": \"object\"}}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"particles\": {\"type\": \"array\"}, \"type\": {\"enum\": [\"Simple\"], \"type\": \"string\"}, \"dimensions\": {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"minimum\": 0, \"type\": \"number\"}}, \"temperature\": {\"minimum\": 0, \"type\": \"number\"}, \"skin_thickness\": {\"minimum\": 0, \"type\": \"number\"}, \"pack\": {\"type\": \"object\", \"properties\": {\"density\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"composition\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"minimum\": 0.0, \"maximum\": 1.0, \"exclusiveMinimum\": true, \"type\": \"number\"}}, \"type\": \"object\"}, \"count\": {\"minimum\": 1, \"type\": \"integer\"}}}, \"nlist_cutoff\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"chemical_potential\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"type\": \"number\"}}, \"type\": \"object\"}}}, \"type\": \"array\", \"minItems\": 1}";
	std::string SAPHRON::JsonSchema::SimpleWorld = "{\"required\": [\"type\", \"dimensions\", \"nlist_cutoff\", \"skin_thickness\", \"components\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"components\": {\"components\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"required\": [\"count\"], \"type\": \"object\", \"properties\": {\"count\": {\"minimum\": 1, \"type\": \"integer\"}}}}, \"additionalProperties\": false, \"minProperties\": 1, \"type\": \"object\"}}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"particles\": {\"type\": \"array\"}, \"type\": {\"enum\": [\"Simple\"], \"type\": \"string\"}, \"dimensions\": {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"minimum\": 0, \"type\": \"number\"}}, \"temperature\": {\"minimum\": 0, \"type\": \"number\"}, \"skin_thickness\": {\"minimum\": 0, \"type\": \"number\"}, \"pack\": {\"type\": \"object\", \"properties\": {\"density\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"composition\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"minimum\": 0.0, \"maximum\": 1.0, \"exclusiveMinimum\": true, \"type\": \"number\"}}, \"type\": \"object\"}, \"count\": {\"minimum\": 1, \"type\": \"integer\"}}}, \"nlist_cutoff\": {\"minimum\": 0, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"chemical_potential\": {\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"type\": \"number\"}}, \"type\": \"object\"}}}";
	std::string SAPHRON::JsonSchema::Particles = "{\"minItems\": 1, \"items\": {\"maxItems\": 4, \"minItems\": 3, \"items\": [{\"minimum\": 1, \"type\": \"integer\"}, {\"type\": \"string\"}, {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"minimum\": 0, \"type\": \"number\"}}, {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"type\": \"number\"}}, {\"type\": \"string\"}], \"type\": \"array\", \"additionalItems\": false}, \"type\": \"array\", \"additionalItems\": false}";
	std::string SAPHRON::JsonSchema::Site = "{\"maxItems\": 4, \"minItems\": 3, \"items\": [{\"minimum\": 1, \"type\": \"integer\"}, {\"type\": \"string\"}, {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"minimum\": 0, \"type\": \"number\"}}, {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"type\": \"number\"}}, {\"type\": \"string\"}], \"type\": \"array\", \"additionalItems\": false}";
	std::string SAPHRON::JsonSchema::Blueprints = "{\"patternProperties\": {\"^[A-z][A-z0-9]+$\": {\"required\": [\"count\"], \"additionalProperties\": false, \"minProperties\": 1, \"type\": \"object\", \"properties\": {\"charge\": {\"type\": \"number\"}, \"bonds\": {\"items\": {\"maxItems\": 2, \"items\": {\"minimum\": 0, \"type\": \"number\"}, \"type\": \"array\", \"minItems\": 2}, \"type\": \"array\"}, \"children\": {\"items\": {\"required\": [\"species\"], \"type\": \"object\", \"properties\": {\"mass\": {\"minimum\": 0, \"type\": \"number\"}, \"species\": {\"type\": \"string\"}, \"charge\": {\"type\": \"number\"}}}, \"type\": \"array\", \"minItems\": 1}, \"mass\": {\"minimum\": 0, \"type\": \"number\"}, \"count\": {\"minimum\": 1, \"type\": \"integer\"}}}}, \"additionalProperties\": false, \"minProperties\": 1, \"type\": \"object\"}";
	std::string SAPHRON::JsonSchema::Selector = "{}";
	std::string SAPHRON::JsonSchema::Director = "{\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"type\": \"number\"}}";
	std::string SAPHRON::JsonSchema::Observers = "{\"type\": \"array\"}";
	std::string SAPHRON::JsonSchema::XYZObserver = "{\"required\": [\"type\", \"prefix\", \"frequency\"], \"additionalProperties\": false, \"properties\": {\"frequency\": {\"minimum\": 1, \"type\": \"integer\"}, \"prefix\": {\"type\": \"string\"}, \"type\": {\"enum\": [\"XYZ\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::JSONObserver = "{\"required\": [\"type\", \"prefix\", \"frequency\"], \"additionalProperties\": false, \"properties\": {\"frequency\": {\"minimum\": 1, \"type\": \"integer\"}, \"prefix\": {\"type\": \"string\"}, \"type\": {\"enum\": [\"JSON\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::DLMFileObserver = "{\"required\": [\"type\", \"prefix\", \"frequency\", \"flags\"], \"additionalProperties\": false, \"properties\": {\"flags\": {\"type\": \"object\", \"properties\": {\"particle_director\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_tensor\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pzz\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_interelect\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_interval\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"dos_flatness\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pxy\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_values\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_bin_count\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"iteration\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pyy\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_volume\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_position\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_lower_outliers\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_intervdw\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_species_id\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_intravdw\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"histogram\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_counts\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"move_acceptances\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_bonded\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"simulation\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_ideal\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_temperature\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_tail\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pxz\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_density\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pyz\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_neighbors\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"dos_factor\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_parent_id\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_id\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_energy\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_composition\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"hist_upper_outliers\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_species\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_chem_pot\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_connectivity\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"world_pressure\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"pressure_pxx\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_intraelect\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"particle_parent_species\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}, \"energy_components\": {\"minimum\": 0, \"maximum\": 1, \"type\": \"integer\"}}}, \"frequency\": {\"minimum\": 1, \"type\": \"integer\"}, \"delimiter\": {\"type\": \"string\"}, \"extension\": {\"type\": \"string\"}, \"type\": {\"enum\": [\"DLMFile\"], \"type\": \"string\"}, \"fixedwmode\": {\"type\": \"boolean\"}, \"prefix\": {\"type\": \"string\"}, \"colwidth\": {\"minimum\": 1, \"type\": \"integer\"}}}";
	std::string SAPHRON::JsonSchema::DeleteParticleMove = "{\"required\": [\"type\", \"species\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"species\": {\"minimumItems\": 1, \"items\": {\"type\": \"string\"}, \"type\": \"array\"}, \"type\": {\"enum\": [\"DeleteParticle\"], \"type\": \"string\"}, \"op_prefactor\": {\"tyoe\": \"boolean\"}}}";
	std::string SAPHRON::JsonSchema::InsertParticleMove = "{\"required\": [\"type\", \"stash_count\", \"species\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"type\": {\"enum\": [\"InsertParticle\"], \"type\": \"string\"}, \"stash_count\": {\"minimum\": 1, \"type\": \"integer\"}, \"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"species\": {\"minimumItems\": 1, \"items\": {\"type\": \"string\"}, \"type\": \"array\"}, \"op_prefactor\": {\"tyoe\": \"boolean\"}}}";
	std::string SAPHRON::JsonSchema::VolumeSwapMove = "{\"required\": [\"type\", \"dv\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"dv\": {\"minimum\": 0, \"type\": \"number\"}, \"type\": {\"enum\": [\"VolumeSwap\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::RotateMove = "{\"required\": [\"type\", \"maxangle\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"maxangle\": {\"minimum\": 0, \"maximum\": 6.283185307179586, \"exclusiveMinimum\": true, \"type\": \"number\"}, \"type\": {\"enum\": [\"Rotate\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::TranslateMove = "{\"required\": [\"type\", \"dx\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"dx\": {\"minimum\": 0, \"type\": \"number\"}, \"type\": {\"enum\": [\"Translate\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::DirectorRotateMove = "{\"required\": [\"type\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"DirectorRotate\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::SpeciesSwapMove = "{\"required\": [\"type\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"SpeciesSwap\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::ParticleSwapMove = "{\"required\": [\"type\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"ParticleSwap\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::Moves = "{\"type\": \"array\"}";
	std::string SAPHRON::JsonSchema::RandomIdentityMove = "{\"required\": [\"type\", \"identities\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"identities\": {\"uniqueItems\": true, \"items\": {\"type\": \"string\"}, \"type\": \"array\", \"minIems\": 1}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"RandomIdentity\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::FlipSpinMove = "{\"required\": [\"type\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"weight\": {\"minimum\": 1, \"type\": \"integer\"}, \"seed\": {\"minimum\": 0, \"type\": \"integer\"}, \"type\": {\"enum\": [\"FlipSpin\"], \"type\": \"string\"}}}";
	std::string SAPHRON::JsonSchema::P2SAConnectivity = "{\"required\": [\"type\", \"coefficient\", \"director\", \"selector\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"selector\": {}, \"director\": {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"type\": \"number\"}}, \"type\": {\"enum\": [\"P2SA\"], \"type\": \"string\"}, \"coefficient\": {\"type\": \"number\"}}}";
	std::string SAPHRON::JsonSchema::Connectivities = "{\"items\": {\"oneOf\": [{\"required\": [\"type\", \"coefficient\", \"director\", \"selector\"], \"additionalProperties\": false, \"type\": \"object\", \"properties\": {\"selector\": {}, \"director\": {\"maxItems\": 3, \"additionalItems\": false, \"minItems\": 3, \"type\": \"array\", \"items\": {\"type\": \"number\"}}, \"type\": {\"enum\": [\"P2SA\"], \"type\": \"string\"}, \"coefficient\": {\"type\": \"number\"}}}]}, \"type\": \"array\"}";
	
}