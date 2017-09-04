/* Based on particle_filter.cpp
 * Created on: Dec 12, 2016. 
 * Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 15;
	is_initialized = true;

	weights.resize(num_particles);

	particles.resize(num_particles);

	// This line creates a normal (Gaussian) distribution for x, y, theta.
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i<num_particles; i++){
        weights[i] = 1.;
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = weights[i];
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    double delta_x;
    double delta_y;
    double theta;
    double v_yawrate = velocity / yaw_rate;

    // This line creates a normal (Gaussian) distribution for x, y, theta.
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0., std_pos[0]);
    std::normal_distribution<double> dist_y(0., std_pos[1]);
    std::normal_distribution<double> dist_theta(0., std_pos[2]);

	for(int i=0; i<num_particles; i++){

	    if (fabs(yaw_rate)>0.001){
            theta = particles[i].theta + yaw_rate * delta_t;
            delta_x = v_yawrate * (sin(theta) - sin(particles[i].theta));
            delta_y = v_yawrate * (-cos(theta) + cos(particles[i].theta));
	    }else{
	        theta = particles[i].theta;
	        delta_x = velocity*cos(theta)*delta_t;
            delta_y = velocity*sin(theta)*delta_t;
	    }

        particles[i].x += delta_x + dist_x(gen);
        particles[i].y += delta_y + dist_y(gen);
        particles[i].theta = theta + dist_theta(gen);
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    int num_landmarks = map_landmarks.landmark_list.size();
	int num_observations = observations.size();

    double x, y, theta;

	const double mG_const = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	const double x_den = 2 * std_landmark[0] * std_landmark[0];
    const double y_den = 2 * std_landmark[1] * std_landmark[1];

    for (int i=0; i<num_particles; i++){ //Iterate over the particles space

        double new_weight = 1.;

        x = particles[i].x;
        y = particles[i].y;
        theta = particles[i].theta;

        for (int j=0; j<num_observations; j++){ //Iterate over the observation space

            double x_map, y_map;
            x_map= x + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
            y_map= y + sin(theta) * observations[j].x + cos(theta) * observations[j].y;

            // Find nearest landmark
            std::vector<Map::single_landmark_s> landmarks_ = map_landmarks.landmark_list;
            std::vector<double> dist_obs2landmark(num_landmarks);
            for (int k=0; k<num_landmarks; k++){ //Iterate over the landmarks space

                double dist_part2landmark = dist(x, y, landmarks_[k].x_f, landmarks_[k].y_f); //Calculate distance from the particle to landmarks
                if (dist_part2landmark < sensor_range) { //Calculate distance from the observations to landmarks
                    dist_obs2landmark[k] = dist(x_map, y_map, landmarks_[k].x_f, landmarks_[k].y_f);
                } else {
                    dist_obs2landmark[k] = sensor_range * sensor_range;
                }
            }

            int min_pos = distance(dist_obs2landmark.begin(),min_element(dist_obs2landmark.begin(),dist_obs2landmark.end()));
            float miu_x = landmarks_[min_pos].x_f;
            float miu_y = landmarks_[min_pos].y_f;

            double x_diff = x_map - miu_x;
            double y_diff = y_map - miu_y;
            double exp_arg = x_diff * x_diff / x_den + y_diff * y_diff / y_den;
            new_weight *= mG_const * exp(-exp_arg);
        }

        particles[i].weight = new_weight;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {

	std::default_random_engine gen; gen.seed(123);
    std::discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> particles_;

    particles_.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        particles_[i] = particles[dist_particles(gen)];
    }
    particles = particles_;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
