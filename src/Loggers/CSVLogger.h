#pragma once

#include "Logger.h"
#include <fstream>
#include <iostream>

namespace Loggers
{
	class CSVLogger : public Logger
	{
		private:
			std::ofstream _modelfs;
			std::ofstream _sitefs;

		protected:
			void LogModelPropertiesInternal(BaseModel& model)
			{
				_modelfs << this->GetCallCount() << ",";
				int i = this->ModelProps.size();
				for(auto &prop : this->ModelProps)
					_modelfs << prop.second(model)
					         << ((--i > 0) ? "," : "\n");
			}

			void LogSitePropertiesInternal(Site& site, bool end)
			{
				_sitefs << this->GetCallCount() << ",";

				// Write site properties.
				int i = 0, m = this->SiteProps.size();
				for(auto &prop : this->SiteProps)
				{
					_sitefs << prop.second(site)
					        << ((i < m - 1) ? "," : "\n");
					i++;
				}
			}

		public:
			CSVLogger(std::string modelfile,
			          std::string sitefile,
			          int frequency = 1)
				: Logger(frequency), _modelfs(modelfile),
				  _sitefs(sitefile)
			{
			}

			// Write headers to the beginning of the file.
			void WriteHeaders()
			{
				// Iteration
				_modelfs << "Iteration,";
				_sitefs <<  "Iteration,";

				// Model properties
				int i = this->ModelProps.size();
				for(auto &prop : this->ModelProps)
					_modelfs << prop.first << ((--i > 0) ? "," : "\n");

				// Site properties.
				i = this->SiteProps.size();
				for(auto &prop : this->SiteProps)
					_sitefs << prop.first << ((--i > 0) ? "," : "\n");
				;
			}

			void FlushLog(){}

			~CSVLogger()
			{
				_modelfs.close();
				_sitefs.close();
			}
	};
}
