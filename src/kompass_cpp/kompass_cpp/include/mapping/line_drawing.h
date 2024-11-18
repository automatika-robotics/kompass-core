#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

inline void bresenham(Eigen::Vector2i p1, Eigen::Vector2i p2,
                      std::vector<Eigen::Vector2i> &points) {
  /*Bresenham line drawing algorithm to generate a set of points on a line
   * connecting two provided points*/

  int x = p1[0], y = p1[1]; // the line points
  int dx = p2[0] - p1[0];
  int dy = p2[1] - p1[1];

  points.emplace_back(x, y);

  // Determine step sizes for x and y based on direction
  int xstep = (dx >= 0) ? 1 : -1;
  int ystep = (dy >= 0) ? 1 : -1;

  // Work with absolute values of differences
  dx = std::abs(dx);
  dy = std::abs(dy);

  int ddy = 2 * dy; // work with double values for full precision
  int ddx = 2 * dx;

  if (ddx >= ddy) { // first octant (0 <= slope <= 1)
    // Initialize the error variables
    int error = dx;

    for (int i = 0; i < dx; i++) { // do not use the first point (already done)
      x += xstep;
      error += ddy;
      if (error > ddx) { // increment y if AFTER the middle
        y += ystep;
        error -= ddx;
      }
      points.emplace_back(x, y); // Add the current point
    }
  } else { // second octant (slope > 1)
    int error = dy;

    for (int i = 0; i < dy; i++) {
      y += ystep;
      error += ddx;
      if (error > ddy) {
        x += xstep;
        error -= ddy;
      }
      points.emplace_back(x, y); // Add the current point
    }
  }
}

inline void bresenhamEnhanced(Eigen::Vector2i p1, Eigen::Vector2i p2,
                              std::vector<Eigen::Vector2i> &points) {
  /* Enhanced Bresenham algorithm to generate a set of points on a super-cover
   * line connecting two provided points*/
  /*based on http://eugen.dedu.free.fr/projects/bresenham/ */

  int x = p1[0], y = p1[1]; // the line points
  int dx = p2[0] - p1[0];
  int dy = p2[1] - p1[1];

  points.emplace_back(x, y);

  // Determine step sizes for x and y based on direction
  int xstep = (dx >= 0) ? 1 : -1;
  int ystep = (dy >= 0) ? 1 : -1;

  // Work with absolute values of differences
  dx = std::abs(dx);
  dy = std::abs(dy);

  int ddy = 2 * dy; // work with double values for full precision
  int ddx = 2 * dx;

  if (ddx >= ddy) { // first octant (0 <= slope <= 1)
    // Initialize the error variables
    int errorprev = dx;
    int error = dx;

    for (int i = 0; i < dx; i++) { // do not use the first point (already done)
      x += xstep;
      error += ddy;
      if (error > ddx) { // increment y if AFTER the middle
        y += ystep;
        error -= ddx;
        if (error + errorprev < ddx) {
          points.emplace_back(x, y - ystep);
        } else if (error + errorprev > ddx) {
          points.emplace_back(x - xstep, y);
        } else {
          points.emplace_back(x - xstep, y);
          points.emplace_back(x, y - ystep);
        }
      }
      points.emplace_back(x, y); // Add the current point
      errorprev = error;
    }
  } else { // second octant (slope > 1)
    int errorprev = dy;
    int error = dy;

    for (int i = 0; i < dy; i++) {
      y += ystep;
      error += ddx;
      if (error > ddy) {
        x += xstep;
        error -= ddy;
        if (error + errorprev < ddy) {
          points.emplace_back(x - xstep, y);
        } else if (error + errorprev > ddy) {
          points.emplace_back(x, y - ystep);
        } else {
          points.emplace_back(x - xstep, y);
          points.emplace_back(x, y - ystep);
        }
      }
      points.emplace_back(x, y); // Add the current point
      errorprev = error;
    }
  }
}

inline void hutchison(Eigen::Vector2i p1, Eigen::Vector2i p2,
                      std::vector<Eigen::Vector2i> &points) {

  const int dy = p2[1] - p1[1];
  const int dx = p2[0] - p1[0];

  // X Major part
  if (std::abs(dx) >= std::abs(dy)) {

    int m = (dx << 16) / dy; // Fixed-point slope

    int x = p1[0] << 16;
    int y = p1[1];

    // Adjust ox to ensure the initial run is shorter
    int ox = x + ((m + (3 << 15)) >> 1); // ox = x + (m + 1.5) / 2

    int row = p1[0];

    // Loop over each scanline except the last one
    int rlen;
    for (; y < p2[1]; y++) {
      x += m;
      rlen = ((x >> 16) - (ox >> 16));
      std::cout << rlen << '\n';
      for (int j = 0; j < rlen; ++j) {
        points.emplace_back(row, y);
        row += 1;
      }
      ox = x;
    }

    // Draw the last run separately
    rlen = p2[0] - (x >> 16) + 1;
    for (int i = 0; i < rlen; ++i) {
      points.emplace_back(p2[0] + i, p2[1]);
    }
  }
  // Y Major part
  else {

    int m = (dy << 16) / dx; // Fixed-point slope

    int y = p1[1] << 16;
    int x = p1[0];

    // Adjust oy to ensure the initial run is shorter
    int oy = y + ((m + (3 << 15)) >> 1); // oy = y + (m + 1.5) / 2

    int col = p1[1];

    // Loop over each scanline except the last one
    int rlen;
    for (; x < p2[0]; ++x) {
      y += m;
      rlen = ((y >> 16) - (oy >> 16));
      for (int j = 0; j < rlen; ++j) {
        points.emplace_back(x, col);
        col += 1;
      }
      oy = y;
    }

    // Draw the last run separately
    rlen = p2[1] - (y >> 16) + 1;
    for (int i = 0; i < rlen; ++i) {
      points.emplace_back(p2[0], p2[1] + i);
    }
  }
}
