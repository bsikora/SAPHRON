#pragma once

#include "Visitor.h"

namespace SAPHRON
{
	// Abstract base class for visitable objects, traversed (usually) by loggers.
	class Visitable
	{
		public:
			virtual void AcceptVisitor(Visitor &v) const = 0;
	};
}
