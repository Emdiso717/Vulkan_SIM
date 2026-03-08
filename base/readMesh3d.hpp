#ifndef READ_MESH_3D_HPP_
#define READ_MESH_3D_HPP_
#include "vtkio.hpp"
#include <Eigen/Eigen>
#include <map>
#include <ranges>
#include <string_view>

namespace io {
namespace internal {
inline bool extractBoundary(const Eigen::MatrixXd &vertices,
                            const Eigen::MatrixXi &tetrahedrons,
                            Eigen::MatrixXi &triangles);
}

inline void readMesh3d(std::string_view filename, Eigen::MatrixXd &V,
                       Eigen::MatrixXi &tets) {
  VtkInput in(filename);
  in.readMesh<VtkCellType::TETRA>(V, tets);
}

inline void readMesh3d(std::string_view filename, Eigen::MatrixXd &V,
                       Eigen::MatrixXi &tets, Eigen::MatrixXi &tris) {
  VtkInput in(filename);
  in.readMesh<VtkCellType::TETRA>(V, tets);
  internal::extractBoundary(V, tets, tris);
}

namespace internal {

inline bool extractBoundary(const Eigen::MatrixXd &vertices,
                            const Eigen::MatrixXi &tetrahedrons,
                            Eigen::MatrixXi &triangles) {
  using Face = std::array<int, 3>;
  using Vec3S = Eigen::Vector3d;
  using Scalar = double;
  using Index = int;

  struct FaceData {
    Face face{};  // Unsorted version (used to preserve orientation)
    int tetId{};  // int of the owning tetrahedron
    int oppVid{}; // Opposite vertex int in the tetrahedron
    bool isUnique{true};
    // Flag to indicate if the face is unique (boundary face)
  };

  struct FaceKey {
    Face sorted;

    bool operator<(const FaceKey &other) const {
      return std::ranges::lexicographical_compare(sorted, other.sorted);
    }
  };

  std::map<FaceKey, FaceData> faceMap;
  const auto &T = tetrahedrons;

  // ---- build face table ----
  for (int i = 0; i < T.rows(); ++i) {
    const int v0 = T(i, 0);
    const int v1 = T(i, 1);
    const int v2 = T(i, 2);
    const int v3 = T(i, 3);

    auto add_face = [&](int a, int b, int c, int opp) {
      Face f = {a, b, c};
      FaceKey key{{a, b, c}};
      std::ranges::sort(key.sorted);
      // faceMap[key].push_back({f, i, opp});
      if (auto it = faceMap.find(key); it != faceMap.end()) {
        it->second.isUnique = false; // Mark as non-unique
      } else {
        faceMap[key] = {f, i, opp, true};
      }
    };

    add_face(v0, v1, v2, v3);
    add_face(v0, v1, v3, v2);
    add_face(v0, v2, v3, v1);
    add_face(v1, v2, v3, v0);
  }

  std::vector<Face> boundary_faces;

  // ---- find boundary faces ----
  for (const auto &f : faceMap | std::views::values) {
    if (!f.isUnique) // not a boundary face
      continue;

    // const auto& tet = tetrahedrons.row(f.tetId);

    // ---- determine orientation ----
    const Vec3S A = vertices.row(f.face[0]);
    const Vec3S B = vertices.row(f.face[1]);
    const Vec3S C = vertices.row(f.face[2]);
    const Vec3S D = vertices.row(f.oppVid);

    // signed volume of tetrahedron ABCD
    const Scalar vol = (B - A).cross(C - A).dot(D - A);
    // if vol > 0, then D is on the left side of face ABC,
    // we want the normal to point outward, so we need to flip the face
    if (vol > 0)
      boundary_faces.push_back({f.face[0], f.face[2], f.face[1]});
    else
      boundary_faces.push_back(f.face);
  }

  if (boundary_faces.empty())
    return false;

  // ---- output result ----
  triangles.resize(static_cast<long long>(boundary_faces.size()), 3);
  for (size_t i = 0; i < boundary_faces.size(); ++i) {
    triangles.row(i) << boundary_faces[i][0], boundary_faces[i][1],
        boundary_faces[i][2];
  }

  return true;
}
} // namespace internal
} // namespace io

#endif