
// STL
#include <vector>

// MATPLOTLIB
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

void line_plot_example()
{
  plt::plot({1, 3, 2, 4});
  plt::show();
}

void legend_example()
{
  // Prepare data.
  int n = 5000;
  std::vector<double> x(n), y(n), z(n), w(n, 2);
  for (int i = 0; i < n; ++i)
  {
    x.at(i) = i * i;
    y.at(i) = sin(2 * M_PI * i / 360.0);
    z.at(i) = log(i);
  }

  // Set the size of output image to 1200x780 pixels
  plt::figure_size(1200, 780);
  // Plot line from given x and y data. Color is selected automatically.
  plt::plot(x, y);
  // Plot a red dashed line from given x and y data.
  plt::plot(x, w, "r--");
  // Plot a line whose name will show up as "log(x)" in the legend.
  plt::named_plot("log(x)", x, z);
  // Set x-axis to interval [0,1000000]
  plt::xlim(0, 1000 * 1000);
  // Add graph title
  plt::title("Sample figure");
  // Enable legend.
  plt::legend();
}

void surface_plot_example()
{
  std::vector<std::vector<double>> x, y, z;
  for (double i = -5; i <= 5; i += 0.25)
  {
    std::vector<double> x_row, y_row, z_row;
    for (double j = -5; j <= 5; j += 0.25)
    {
      x_row.push_back(i);
      y_row.push_back(j);
      z_row.push_back(::std::sin(::std::hypot(i, j)));
    }
    x.push_back(x_row);
    y.push_back(y_row);
    z.push_back(z_row);
  }

  plt::plot_surface(x, y, z);
  plt::show();
}

int main(int, char **)
{
  std::cout << __FILE__ << std::endl;

  line_plot_example();

  legend_example();

  surface_plot_example();

  return 0;
}