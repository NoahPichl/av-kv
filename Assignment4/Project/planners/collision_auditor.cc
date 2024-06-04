
#include <math.h>
#include <array>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>




std::vector<bool> collision_checker(
    const State& ego_vehicle,
    const std::vector<State>& obstacles
) {
    std::vector<bool> collisions(obstacles.size(), false);
    auto n_circles = CIRCLE_OFFSETS.size();
    float cos_yaw = std::cos(ego_vehicle.rotation.yaw);
    float sin_yaw = std::sin(ego_vehicle.rotation.yaw);

    for (size_t obs = 0; obs < obstacles.size(); ++obs) {
        auto actor = obstacles[obs];
        bool collision = false;

        for (size_t c = 0; c < n_circles && !collision; ++c) {
            float ego_offset_cos = CIRCLE_OFFSETS[c] * cos_yaw;
            float ego_offset_sin = CIRCLE_OFFSETS[c] * sin_yaw;

            for (size_t c2 = 0; c2 < n_circles && !collision; ++c2) {
                float actor_offset_cos = CIRCLE_OFFSETS[c2] * std::cos(actor.rotation.yaw);
                float actor_offset_sin = CIRCLE_OFFSETS[c2] * std::sin(actor.rotation.yaw);

                double dist = std::hypot(
                    ego_vehicle.location.x + ego_offset_cos - actor.location.x - actor_offset_cos,
                    ego_vehicle.location.y + ego_offset_sin - actor.location.y - actor_offset_sin
                );

                collision = dist < (CIRCLE_RADII[c] + CIRCLE_RADII[c2]);
            }
        }
        collisions[obs] = collision;
    }

    return collisions;
}
}