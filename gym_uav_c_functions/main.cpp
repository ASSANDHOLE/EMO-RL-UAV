#include <bits/stdc++.h>

#define p std::make_pair
#define t Direction
#define s1 1.0f
#define s2 1.4142f
#define TO_IDX(a) ((a).first * len + (a).second)
#define TO_IDX2(a, b) ((a) * len + (b))

struct Direction {
    int x;
    int y;
    float val;
};

inline bool Valid(int x, int y, uint32_t len) {
    return 0 <= x && x < len && 0 <= y && y < len;
}

extern "C" int Dijkstra(const uint8_t *arr, uint32_t len, int init_pos_x, int init_pos_y) {
    auto m_len = len * len;
    std::vector<bool> finalized(m_len, false);
    std::vector<float> dist(m_len, float(INT32_MAX));
    dist[TO_IDX2(init_pos_x, init_pos_y)] = 0;
    typedef std::pair<std::pair<int, int>, float> POS_DEST;
    auto cmp = [](const POS_DEST &a, const POS_DEST &b) { return a.second > b.second; };
    std::priority_queue<POS_DEST, std::vector<POS_DEST>, decltype(cmp)> pos_dest(cmp);
    std::array<Direction, 8> directions{
            t{-1, -1, s2}, t{-1, 0, s1}, t{-1, 1, s2}, t{0, -1, s1},
            t{0, 1, s1}, t{1, -1, s2}, t{1, 0, s1}, t{1, 1, s2}
    };
    pos_dest.emplace(p(init_pos_x, init_pos_y), 0);
    while (!pos_dest.empty()) {
        auto cur_pd = pos_dest.top();
        auto cur_pos = cur_pd.first;
        auto pos = TO_IDX(cur_pos);
        pos_dest.pop();
        for (auto &direction: directions) {
            int n_pos_x = cur_pos.first + direction.x;
            int n_pos_y = cur_pos.second + direction.y;
            auto n_pos = TO_IDX2(n_pos_x, n_pos_y);
            if (Valid(n_pos_x, n_pos_y, len) && arr[n_pos] != 1 && !finalized[n_pos]) {
                if (dist[n_pos] > dist[pos] + direction.val) {
                    dist[n_pos] = dist[pos] + direction.val;
                    pos_dest.emplace(p(n_pos_x, n_pos_y), dist[n_pos]);
                }
            }
        }
    }
    for (int i = 0; i < m_len; ++i) {
        if (arr[i] == 2) {
            return int(dist[i]);
        }
    }
    return -1;
}

extern "C" double GetDistance(const double *obs_x, const double *obs_y, double uav_x, double uav_y, double end_x, double end_y, double r,
                       int obs_cnt) {
    double delta_y = end_y - uav_y;
    double delta_x = end_x - uav_x;
    double line_length = pow(pow(delta_x, 2) + pow(delta_y, 2), 0.5);
    double cos = delta_x / line_length;
    double sin = delta_y / line_length;
    double dist = line_length;
    for (int i = 0; i < obs_cnt; i++) {
        double foot_x =
                (pow(delta_y, 2) * uav_x + pow(delta_x, 2) * obs_x[i] + (obs_y[i] - uav_y) * (delta_x) * (delta_y)) /
                pow(line_length, 2);
        double dist_to_line = (fabs(sin) < 0.0001) ? obs_y[i] - uav_y : fabs((obs_x[i] - foot_x) / sin);
        if (dist_to_line <= r) {
            double d = pow(pow(r, 2) - pow(dist_to_line, 2), 0.5);
            if (fabs(cos) < 0.0001) {
                double intersect_y1 = obs_y[i] - d;
                double intersect_y2 = obs_y[i] + d;
                if ((intersect_y1 >= uav_y && intersect_y1 <= end_y) ||
                    (intersect_y1 <= uav_y && intersect_y2 >= end_y)) {
                    double tmp_dist = fabs(intersect_y1 - uav_y);
                    dist = fmin(tmp_dist, dist);
                }
                if ((intersect_y2 >= uav_y && intersect_y2 <= end_y) ||
                    (intersect_y2 <= uav_y && intersect_y2 >= end_y)) {
                    double tmp_dist = fabs(intersect_y2 - uav_y);
                    dist = fmin(tmp_dist, dist);
                }
            }
            else {
                double intersect_x1 = foot_x - d * cos;
                double intersect_x2 = foot_x + d * cos;
                if ((intersect_x1 >= uav_x && intersect_x1 <= end_x) ||
                    (intersect_x1 <= uav_x && intersect_x2 >= end_x)) {
                    double tmp_dist = fabs((intersect_x1 - uav_x) / cos);
                    dist = fmin(tmp_dist, dist);
                }
                if ((intersect_x2 >= uav_x && intersect_x2 <= end_x) ||
                    (intersect_x2 <= uav_x && intersect_x2 >= end_x)) {
                    double tmp_dist = fabs((intersect_x2 - uav_x) / cos);
                    dist = fmin(tmp_dist, dist);
                }
            }
        }
    }
    return dist;
}
