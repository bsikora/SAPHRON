#pragma once

#include "../DataLoggers/DataLogger.h"
#include "../Histogram.h"
#include "../Models/BaseModel.h"
#include "../Rand.h"
#include "Ensemble.h"
#include <iomanip>
#include <vector>

namespace Ensembles
{
	typedef std::pair<double, double> Interval;

	// Abstract class for density of states based sampling using flat histogram
	// method.
	template<typename T>
	class DensityOfStatesEnsemble : public Ensemble<T>
	{
		private:

			// Flatness threshold.
			double _targetFlatness = 0.8;

			// Densities of state pointer.
			std::vector<double>* _DOS;

			// Histogram counts.
			std::vector<double>* _counts;

			// Flatness measure.
			double _flatness;

			// Lower outliers.
			double _lowerOutliers = 0;

			// Upper outliers.
			double _upperOutliers = 0;

			// Log of scaling factor for density of states.
			double _scaleFactor = 1;

			// Parameter histogram interval.
			Interval _interval;

			// Unique identifier of walker number.
			double _walker;

		protected:
			// Random number generator.
			Rand rand;

			// Histogram
			Histogram hist;

			// Current energy of the model.
			double _energy;

			// Calculates the total energy of the model.
			double CalculateTotalEnergy();

			// Registers loggable properties with logger.
			void RegisterLoggableProperties(DataLogger& logger);

		public:

			// Initializes a DOS ensemble for a specified model. The binning will
			// be performed according to the specified minimum and maxiumum parameter and
			// the bin count.
			DensityOfStatesEnsemble(BaseModel& model,
			                        double minP,
			                        double maxP,
			                        int binCount) :
				Ensemble<T>(model), rand(15), hist(minP, maxP, binCount)
			{
				_interval = Interval(minP, maxP);

				_counts = hist.GetHistogramPointer();
				_DOS = hist.GetValuesPointer();

				// Calculate initial energy.
				_energy = CalculateTotalEnergy();

				std::cout << "Density of States Ensemble Initialized." << std::endl;
				std::cout << "Interval: " << std::setw(10) << std::left << minP;
				std::cout << std::setw(10) << std::right << maxP << std::endl <<
				std::endl;
			};

			// Runs multiple DOS simulations, between each subsequent iteration is
			// a re-normalization step involving resetting of the hisogram and reduction of
			// the scaling factor.
			virtual void Run(int iterations);

			// Performs one DOS iteration. This constitutes sampling phase space
			// until the density of states histogram is determined to be flat.
			// Performs one Monte-Carlo iteration. This is precisely one random draw
			// from the model (one function call to model->DrawSample()).
			virtual void Sweep();

			// Performs one Monte-Carlo iteration. This is precisely one random draw
			// from the model (one function call to model->DrawSample()).
			virtual void Iterate() = 0;

			// Gets the current energy of the system.
			double GetEnergy()
			{
				return _energy;
			}

			// Gets the density of states scaling factor.
			double GetScaleFactor()
			{
				return _scaleFactor;
			}

			// Sets the density of states scaling factor.
			virtual double SetScaleFactor(double sf)
			{
				return _scaleFactor = sf;
			}

			virtual double GetTargetFlatness()
			{
				return _targetFlatness;
			}

			virtual double SetTargetFlatness(double f)
			{
				return _targetFlatness = f;
			}

			// Resets the histogram.
			void ResetHistogram()
			{
				hist.ResetHistogram();
			}

			// Gets the interval over which the DOS is computed.
			Interval GetParameterInterval()
			{
				return _interval;
			}

			// Reduces the scaling factor order by a specified multiple.
			double ReduceScaleFactor(double order = 0.5)
			{
				// We store log of scale factor. So we simply multiply.
				return _scaleFactor = _scaleFactor*order;
			}

			// Gets the walker ID.
			int GetWalkerID()
			{
				return _walker;
			}

			// Sets the walker ID.
			int SetWalkerID(int id)
			{
				return _walker = id;
			}

			// Acceptance probability based on density of states.
			virtual double AcceptanceProbability(double prevH, double currH) = 0;
	};
}
