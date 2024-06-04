
#include <math.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>



std::vector<State> generate_offset_goals(const State& goal_state) {
    std::vector<State> goals_offset;
    
    const float yaw_plus_90 = goal_state.rotation.yaw + M_PI_2;
    const float cos_yaw_plus_90 = std::cos(yaw_plus_90);
    const float sin_yaw_plus_90 = std::sin(yaw_plus_90);
    
    for (int goal_nr = 0; goal_nr < _num_goals; ++goal_nr) {
        State goal_offset = goal_state;
        
        const float offset_product = (goal_nr - _num_goals / 2) * _goal_offset;
        
        goal_offset.location.x += offset_product * cos_yaw_plus_90;
        goal_offset.location.y += offset_product * sin_yaw_plus_90;
        
        goals_offset.push_back(goal_offset);
    }
    
    return goals_offset;
}