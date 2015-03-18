#pragma once

#include "Neighbor.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace SAPHRON
{
	struct Position
	{
		double x;
		double y;
		double z;

		bool operator==(const Position& rhs)
		{
			return this->x == rhs.x && this->y == rhs.y && this->z == rhs.z;
		}
	};

	typedef std::vector<double> Director;
	typedef std::list<Neighbor> NeighborList;
	typedef std::list<Neighbor>::iterator NeighborIterator;
	typedef std::vector<std::string> IdentityList;

	// Abstract class Particle represents either a composite or primitive object, from an atom/site to
	// a molecule to a collection of molecules. It represents an common interface allowing the manipulation
	// of all of the above through a common interface.
	class Particle
	{
		private:

			// String identifier.
			std::string _identifier;

			// Integer identifier.
			int _ID;

			// Neighbor identifier.
			NeighborList _neighbors;

			// Next ID counter for unique map ID.
			static int _nextID;

			// Global list of particle identities.
			static IdentityList _identityList;

		public:

			// Initialize a particle with a particular identifier. This string represents the global type
			// identifier for this particle.
			Particle(std::string identifier) : _identifier(identifier), _ID(0)
			{
				SetIdentity(identifier);
			}

			virtual ~Particle() {}

			// Get particle identifier.
			int GetIdentifier()
			{
				return _ID;
			}

			// Get particle string identifier.
			std::string GetIdentifierString()
			{
				return _identifier;
			}

			// Get identity list.
			static IdentityList GetIdentityList()
			{
				return _identityList;
			}

			// Set the identity of a particle.
			void SetIdentity(std::string identifier)
			{
				auto search = std::find(_identityList.begin(), _identityList.end(), identifier);
				if (search != _identityList.end())
					_ID = search - _identityList.begin();
				else
				{
					_ID = (int) _identityList.size();
					_identityList.push_back(identifier);
				}

				_identifier = identifier;
			}

			// Set the identity of a particle.
			void SetIdentity(int id)
			{
				assert(id < (int)_identityList.size());
				_ID = id;
				_identifier = _identityList[id];
			}

			// Get particle position.
			virtual Position GetPosition() const = 0;

			// Move a particle to a new set of coordinates.
			virtual void SetPosition(const Position& position) = 0;

			// Move a particle to a new set of coordinates.
			virtual void SetPosition(Position && position) = 0;

			// Get the particle director.
			virtual Director GetDirector() const = 0;

			// Get the particle director pointer.
			virtual const Director& GetDirectorRef() const = 0;

			// Set the particle director.
			virtual void SetDirector(const Director& director) = 0;

			// Set the particle director.
			virtual void SetDirector(Director && director) = 0;

			// Gets neighbor list iterator.
			NeighborList& GetNeighborList()
			{
				return _neighbors;
			}

			// Add a neighbor to neighbor list.
			void AddNeighbor(Neighbor && neighbor)
			{
				_neighbors.emplace_back(neighbor);
			}

			// Gets all descendants of a particle.
			virtual std::vector<Particle*> GetChildren() = 0;

			// Clone particle.
			virtual Particle* Clone() const = 0;
	};

	IdentityList Particle::_identityList;
}
