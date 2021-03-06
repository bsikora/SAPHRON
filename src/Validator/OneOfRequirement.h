#pragma once 

#include "Requirement.h"
#include "RequirementLoader.h"

namespace Json
{
	class OneOfRequirement : public Requirement
	{
	private:
		RequireList _reqs;

	public:
		OneOfRequirement() : _reqs(0) {}

		virtual void ClearErrors() override
		{
			for(auto& r : _reqs)
				r->ClearErrors();

			Requirement::ClearErrors();
		}

		virtual void ClearNotices() override
		{
			for(auto& r : _reqs)
				r->ClearNotices();

			Requirement::ClearNotices();
		} 

		virtual void Reset() override
		{
			ClearErrors();
			ClearNotices();
			_reqs.clear();
		}

		virtual void Parse(Value json, const std::string& path) override
		{
			Reset();
			RequirementLoader loader;

			auto& head = json.isMember("oneOf") ? json["oneOf"] : json;

			for(auto& val : head)
				if(auto req = loader.LoadRequirement(val))
				{
					_reqs.push_back(std::move(req));
					_reqs.back()->Parse(val, path);
				}

		}

		virtual void Validate(const Value& json, const std::string& path) override
		{
			int validated = 0;
			for(auto& r : _reqs)
			{
				r->Validate(json, path);
				if(!r->HasErrors())
					++validated;
			}

			if(validated > 1)
				PushError(path + ": Input must validate against only one schema");
			else if(validated == 0)
				for(auto& r : _reqs)
				{
					if(r->HasErrors())
						for(const auto& error : r->GetErrors())
							PushError(error);
			
					if(r->HasNotices())
						for(const auto& notice : r->GetNotices())
							PushNotice(notice);
				}
		}
	};
}