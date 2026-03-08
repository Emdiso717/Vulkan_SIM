#ifndef VTK_IO_HPP_
#define VTK_IO_HPP_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

enum class VtkCellType : uint8_t {
  VERTEX = 1,
  POLY_VERTEX = 2,
  LINE = 3,
  POLY_LINE = 4,
  TRIANGLE = 5,
  TRIANGLE_STRIP = 6,
  POLYGON = 7,
  PIXEL = 8,
  QUAD = 9,
  TETRA = 10,
  VOXEL = 11,
  HEXAHEDRON = 12,
  WEDGE = 13,
  PYRAMID = 14,
  PENTAGONAL_PRISM = 15,
  HEXAGONAL_PRISM = 16,

  QUADRATIC_EDGE = 21,
  QUADRATIC_TRIANGLE = 22,
  QUADRATIC_QUAD = 23,
  QUADRATIC_TETRA = 24,
  QUADRATIC_HEXAHEDRON = 25,
  QUADRATIC_WEDGE = 26,
  QUADRATIC_PYRAMID = 27,
};

inline constexpr int vtkCellNumIndices[]{
    -1,
    1,  // VERTEX
    -1, // POLY_VERTEX
    2,  // LINE
    -1, // POLY_LINE
    3,  // TRIANGLE
    -1, // TRIANGLE_STRIP
    -1, // POLYGON
    4,  // PIXEL
    4,  // QUAD
    4,  // TETRA
    8,  // VOXEL
    8,  // HEXAHEDRON
    6,  // WEDGE
    5,  // PYRAMID
    10, // PENTAGONAL_PRISM
    12, // HEXAGONAL_PRISM

    3,  // QUADRATIC_EDGE
    6,  // QUADRATIC_TRIANGLE
    8,  // QUADRATIC_QUAD
    10, // QUADRATIC_TETRA
    20, // QUADRATIC_HEXAHEDRON
    15, // QUADRATIC_WEDGE
    13, // QUADRATIC_PYRAMID
};

struct VtkCells {
  std::vector<int> indices;
  std::vector<int> offsets;
  std::vector<VtkCellType> cellTypes;

  template <typename DerivedC>
  void toEigen(const VtkCellType cellType,
               Eigen::PlainObjectBase<DerivedC> &C) const {

    const int stride = vtkCellNumIndices[static_cast<int>(cellType)];
    if (stride == -1)
      throw std::runtime_error("[VtkIO] Unsupported cell type");

    const int nC = cellTypes.size();
    C.resize(nC, stride);
    for (int i = 0; i < nC; ++i) {
#ifndef NDEBUG
      if (cellTypes[i] != cellType)
        throw std::runtime_error("[VtkIO] Cell type mismatch");
#endif

      const int offset = offsets[i];
      for (int j = 0; j < stride; ++j)
        C(i, j) = indices[offset + j];
    }
  }
};

struct VtkAttrib {
  std::string name;
  enum class Type : uint8_t { NONE, POINT, CELL } type;
  enum class Format : uint8_t { SCALAR, VECTOR } format;
  enum class DType : uint8_t { INT, DOUBLE } dtype;

  void setFormat(const std::string_view &fmt) {
    if (fmt == "SCALARS")
      format = Format::SCALAR;
    else if (fmt == "VECTORS")
      format = Format::VECTOR;
    else
      throw std::runtime_error("[VtkIO] Invalid data format");
  }

  void setDataType(const std::string_view &ty) {
    if (ty == "int")
      dtype = DType::INT;
    else if (ty == "double")
      dtype = DType::DOUBLE;
    else
      throw std::runtime_error("[VtkIO] Invalid data type");
  }

  operator bool() const { return type != Type::NONE; }
};

class VtkInput {
private:
  struct Version {
    int major;
    int minor;
  };

  static constexpr int maxMajorVersion = 5;
  static constexpr int minMajorVersion = 1;

public:
  VtkInput(const std::filesystem::path &path)
#if defined(__ANDROID__)
      : m_in(m_assetStream) {
    AAsset *asset = AAssetManager_open(androidApp->activity->assetManager,
                                       path.c_str(), AASSET_MODE_BUFFER);
    if (!asset)
      throw std::runtime_error("[VtkIO] Failed to open asset ");
    const char *data = (const char *)AAsset_getBuffer(asset);
    size_t size = AAsset_getLength(asset);

    std::string buf(data, size);

    AAsset_close(asset);

    m_assetStream.str(std::move(buf));
    m_assetStream.clear();
    m_assetStream.seekg(0);
  }
#else
      : m_fin(path), m_in(m_fin) {
    if (!m_fin.is_open())
      throw std::runtime_error("[VtkIO] Failed to open file " + path.string());
  }
#endif

  VtkInput(std::ifstream &&fin) : m_fin(std::move(fin)), m_in(m_fin) {
    if (!m_fin.is_open())
      throw std::runtime_error("[VtkIO] Failed to open file");
  }

  VtkInput(std::istream &in) : m_in(in) {}

  VtkInput(const VtkInput &) = delete;
  VtkInput &operator=(const VtkInput &) = delete;
  VtkInput(VtkInput &&) = delete;
  VtkInput &operator=(VtkInput &&) = delete;

  template <typename DerivedV>
  VtkInput &readUnstructuredGrid(Eigen::PlainObjectBase<DerivedV> &V,
                                 VtkCells &cells) {

    const auto version = readVersion();
    if (version.major < minMajorVersion || version.major > maxMajorVersion)
      throw std::runtime_error("[VtkIO] Unsupported VTK version");

    std::string buffer;

    std::getline(m_in, buffer); // header

    std::getline(m_in, buffer); // format
      buffer.erase(std::remove(buffer.begin(), buffer.end(), '\r'), buffer.end());
    if (buffer != "ASCII")
      throw std::runtime_error("[VtkIO] Only ASCII format is supported");

    m_in >> buffer; // DATASET
    m_in >> buffer; // UNSTRUCTURED_GRID
      buffer.erase(std::remove(buffer.begin(), buffer.end(), '\r'), buffer.end());
    if (buffer != "UNSTRUCTURED_GRID")
      throw std::runtime_error("[VtkIO] Only UNSTRUCTURED_GRID is supported");

    readVertices(V);
    m_nV = V.rows();

    switch (version.major) {
    case 1:
    case 2:
    case 3:
    case 4:
      readCells_1_4(cells);
      break;
    case 5:
      readCells_5(cells);
      break;
    }
    m_nC = cells.cellTypes.size();

    return *this;
  }

  template <typename DerivedV, typename DerivedC>
  VtkInput &readMesh(const VtkCellType cellType,
                     Eigen::PlainObjectBase<DerivedV> &V,
                     Eigen::PlainObjectBase<DerivedC> &C) {
    VtkCells cells;
    readUnstructuredGrid(V, cells);
    cells.toEigen(cellType, C);
    return *this;
  }

  template <VtkCellType CellType, typename DerivedV, typename DerivedC>
  VtkInput &readMesh(Eigen::PlainObjectBase<DerivedV> &V,
                     Eigen::PlainObjectBase<DerivedC> &C) {
    VtkCells cells;
    readUnstructuredGrid(V, cells);
    cells.toEigen(CellType, C);
    return *this;
  }

  template <typename Derived>
  VtkInput &readPointData(Eigen::PlainObjectBase<Derived> &data,
                          std::string *name = nullptr) {
    const auto attrib = getNextAttrib();
    if (attrib.type != VtkAttrib::Type::POINT)
      throw std::runtime_error("[VtkIO] Point data not found");

    if (name)
      *name = attrib.name;

    readDataArray(attrib, data);

    return *this;
  }

  template <typename Derived>
  VtkInput &readCellData(Eigen::PlainObjectBase<Derived> &data,
                         std::string *name = nullptr) {
    const auto attrib = getNextAttrib();
    if (attrib.type != VtkAttrib::Type::CELL)
      throw std::runtime_error("[VtkIO] Cell data not found");

    if (name)
      *name = attrib.name;

    readDataArray(attrib, data);

    return *this;
  }

  VtkAttrib getNextAttrib() {
    VtkAttrib attrib;
    attrib.type = VtkAttrib::Type::NONE;

    std::string buffer;
    m_in >> buffer;

    if (!m_in)
      return attrib;

    int n;
    std::string attribFormat;
    if (!m_hasReadPointData) {
      m_hasReadPointData = true;
      if (buffer == "POINT_DATA") {
        attrib.type = VtkAttrib::Type::POINT;
        m_in >> n;
        if (n != m_nV)
          throw std::runtime_error("[VtkIO] Point data size mismatch");
        m_in >> attribFormat;

      } else if (buffer == "CELL_DATA") {
        m_hasReadCellData = true;
        attrib.type = VtkAttrib::Type::CELL;
        m_in >> n;
        if (n != m_nC)
          throw std::runtime_error("[VtkIO] Cell data size mismatch");
        m_in >> attribFormat;
      } else {
        throw std::runtime_error("[VtkIO] Invalid data type");
      }
    } else {
      if (buffer == "CELL_DATA") {
        m_hasReadCellData = true;
        attrib.type = VtkAttrib::Type::CELL;
        m_in >> n;
        if (n != m_nC)
          throw std::runtime_error("[VtkIO] Cell data size mismatch");
        m_in >> attribFormat;
      } else {
        attrib.type =
            m_hasReadCellData ? VtkAttrib::Type::CELL : VtkAttrib::Type::POINT;
        attribFormat = buffer;
      }
    }

    std::string attribDType;
    m_in >> attrib.name >> attribDType;
    attrib.setFormat(attribFormat);
    attrib.setDataType(attribDType);

    if (attrib.format == VtkAttrib::Format::SCALAR) {
      std::getline(m_in, buffer);
      std::getline(m_in, buffer);
    }

    return attrib;
  }

  template <typename Derived>
  void readDataArray(const VtkAttrib &attrib,
                     Eigen::PlainObjectBase<Derived> &data) {
    const int n = attrib.type == VtkAttrib::Type::POINT ? m_nV : m_nC;
    const int stride = attrib.format == VtkAttrib::Format::SCALAR ? 1 : 3;
    data.resize(n, stride);
    if (attrib.dtype == VtkAttrib::DType::INT)
      readDataArray<long>(n, stride, data);
    else if (attrib.dtype == VtkAttrib::DType::DOUBLE)
      readDataArray<double>(n, stride, data);
  }

  void skipDataArray(const VtkAttrib &attrib) {
    const int n = attrib.type == VtkAttrib::Type::POINT ? m_nV : m_nC;
    const int stride = attrib.format == VtkAttrib::Format::SCALAR ? 1 : 3;
    std::string buffer;
    for (int i = 0; i < n * stride; ++i)
      m_in >> buffer;
  }

private:
  Version readVersion() {
    std::string buffer;
    std::getline(m_in, buffer);

    auto pos = buffer.find("Version");
    if (pos == std::string::npos)
      throw std::runtime_error("[VtkIO] Failed to determine VTK version");

    std::string versionStr = buffer.substr(pos + 8);
    auto dotPos = versionStr.find('.');
    if (dotPos == std::string::npos)
      throw std::runtime_error("[VtkIO] Failed to determine VTK version");

    Version version;
    version.major = std::stoi(versionStr.substr(0, dotPos));
    version.minor = std::stoi(versionStr.substr(dotPos + 1));
    return version;
  }

  template <typename DerivedV>
  void readVertices(Eigen::PlainObjectBase<DerivedV> &V) {

    std::string buffer;
    int nV;
    m_in >> buffer >> nV >> buffer;

    V.resize(nV, 3);
    for (int i = 0; i < nV; ++i)
      m_in >> V(i, 0) >> V(i, 1) >> V(i, 2);
  }

  void readCells_1_4(VtkCells &cells) {

    std::string buffer;

    int nC, nI;
    m_in >> buffer >> nC >> nI; // CELLS

    cells.indices.resize(nI - nC);
    cells.offsets.resize(nC + 1);

    cells.offsets[0] = 0;
    for (int i = 0; i < nC; ++i) {
      const int offset = cells.offsets[i];

      int n;
      m_in >> n;

      cells.offsets[i + 1] = offset + n;
      for (int j = 0; j < n; ++j)
        m_in >> cells.indices[offset + j];
    }

    readCellTypes(cells.cellTypes);
  }

  void readCells_5(VtkCells &cells) {

    std::string buffer;

    int nC, nI;
    m_in >> buffer >> nC >> nI;
    m_in >> buffer >> buffer; // OFFSETS vtktypeint64

    cells.offsets.resize(nC);
    for (int i = 0; i < nC; ++i)
      m_in >> cells.offsets[i];

    m_in >> buffer >> buffer; // CONNECTIVITY vtktypeint64

    cells.indices.resize(nI);
    for (int i = 0; i < nI; ++i)
      m_in >> cells.indices[i];

    readCellTypes(cells.cellTypes);
  }

  void readCellTypes(std::vector<VtkCellType> &cellTypes) {
    std::string buffer;
    int nC;
    m_in >> buffer >> nC; // CELL_TYPES
    cellTypes.resize(nC);
    for (int i = 0; i < nC; ++i) {
      int cellType;
      m_in >> cellType;
      cellTypes[i] = static_cast<VtkCellType>(cellType);
    }
  }

  template <typename T, typename Derived>
  void readDataArray(const int n, const int stride,
                     Eigen::PlainObjectBase<Derived> &data) {
    data.resize(n, stride);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < stride; ++j) {
        T val;
        m_in >> val;
        data(i, j) = val;
      }
  }

private:
  std::ifstream m_fin;
  std::istringstream m_assetStream;
  std::istream &m_in;

  int m_nV = -1;
  int m_nC = -1;

  bool m_hasReadPointData = false;
  bool m_hasReadCellData = false;
};

class VtkOutput {
public:
  VtkOutput(const std::filesystem::path &path) : m_fout(path), m_out(m_fout) {
    if (!m_fout.is_open())
      throw std::runtime_error("[VtkIO] Failed to open file " + path.string());
  }

  VtkOutput(std::ofstream &&fout) : m_fout(std::move(fout)), m_out(m_fout) {
    if (!m_fout.is_open())
      throw std::runtime_error("[VtkIO] Failed to open file");
  }

  VtkOutput(std::ostream &out) : m_out(out) {}

  VtkOutput(const VtkOutput &) = delete;
  VtkOutput &operator=(const VtkOutput &) = delete;
  VtkOutput(VtkOutput &&) = delete;
  VtkOutput &operator=(VtkOutput &&) = delete;

  VtkOutput &setName(const std::string_view &name) {
    m_name = name;
    return *this;
  }

  VtkOutput &noFullPrecision() {
    m_fullPrecision = false;
    return *this;
  }

  template <typename DerivedV>
  VtkOutput &writeUnstructuredGrid(const Eigen::MatrixBase<DerivedV> &V,
                                   const VtkCells &C) {

    writeHeader();
    writeVertices(V);
    writeCells(C);
    writeCellTypes(C.cellTypes);

    return *this;
  }

  template <typename DerivedV, typename DerivedC>
  VtkOutput &writeMesh(const VtkCellType cellType,
                       const Eigen::MatrixBase<DerivedV> &V,
                       const Eigen::MatrixBase<DerivedC> &C) {

    writeHeader();
    writeVertices(V);
    writeCells(cellType, C);
    writeCellTypes(cellType, C.rows());

    return *this;
  }

  template <VtkCellType CellType, typename DerivedV, typename DerivedC>
  VtkOutput &writeMesh(const Eigen::MatrixBase<DerivedV> &V,
                       const Eigen::MatrixBase<DerivedC> &C) {

    writeHeader();
    writeVertices(V);
    writeCells(CellType, C);
    writeCellTypes<CellType>(C.rows());

    return *this;
  }

  template <typename DerivedV>
  VtkOutput &writeVerticesMesh(const Eigen::MatrixBase<DerivedV> &V) {
    writeHeader();
    writeVertices(V);

    m_nC = V.rows();
    m_out << "CELLS " << V.rows() << " " << 2 * V.rows() << "\n";
    for (int i = 0; i < V.rows(); ++i)
      m_out << "1 " << i << '\n';

    writeCellTypes<VtkCellType::VERTEX>(V.rows());

    return *this;
  }

  template <typename DerivedV>
  VtkOutput &writePolyline(const Eigen::MatrixBase<DerivedV> &V,
                           const bool closed = false) {
    writeHeader();
    writeVertices(V);

    m_nC = std::max<int>(0, closed ? V.rows() : (V.rows() - 1));
    m_out << "CELLS " << m_nC << " " << 3 * m_nC << "\n";
    for (int i = 0; i < m_nC; ++i)
      m_out << "2 " << i << ' ' << ((i + 1) % V.rows()) << '\n';

    writeCellTypes<VtkCellType::LINE>(m_nC);

    return *this;
  }

  template <typename Derived>
  VtkOutput &writeAttrib(const VtkAttrib &attrib,
                         const Eigen::MatrixBase<Derived> &data) {
    if (attrib.type == VtkAttrib::Type::POINT)
      writePointData(attrib.name, data);
    else
      writeCellData(attrib.name, data);

    return *this;
  }

  template <typename Derived>
  VtkOutput &writePointData(const std::string_view &name,
                            const Eigen::MatrixBase<Derived> &data) {
    if (m_hasWrittenCellData)
      throw std::runtime_error(
          "[VtkIO] Point data must be written before cell data");

    if (data.rows() != m_nV)
      throw std::runtime_error("[VtkIO] Point data size mismatch");

    if (!m_hasWrittenPointData) {
      m_out << "POINT_DATA " << data.rows() << '\n';
      m_hasWrittenPointData = true;
    }

    writeDataArray(name, data);

    return *this;
  }

  template <typename Derived>
  VtkOutput &writeCellData(const std::string_view &name,
                           const Eigen::MatrixBase<Derived> &data) {

    if (data.rows() != m_nC)
      throw std::runtime_error("[VtkIO] Cell data size mismatch");

    if (!m_hasWrittenCellData) {
      m_out << "CELL_DATA " << data.rows() << '\n';
      m_hasWrittenCellData = true;
    }

    writeDataArray(name, data);

    return *this;
  }

private:
  void writeHeader() {
    m_out << "# vtk DataFile Version 3.0\n";
    m_out << m_name << '\n';
    m_out << "ASCII\n";
    m_out << "DATASET UNSTRUCTURED_GRID\n";
  }

  template <typename DerivedV>
  void writeVertices(const Eigen::MatrixBase<DerivedV> &V) {
    if (V.cols() != 3)
      throw std::runtime_error("[VtkIO] Vertex data should have 3 columns");

    m_nV = V.rows();
    m_out << "POINTS " << V.rows() << " double\n";
    m_out << V.format(Eigen::IOFormat(
        m_fullPrecision ? Eigen::FullPrecision : Eigen::StreamPrecision,
        Eigen::DontAlignCols, " ", "\n", "", "", "", "\n"));
  }

  void writeCells(const VtkCells &C) {
    m_nC = C.cellTypes.size();

    m_out << "CELLS " << m_nC << " " << (m_nC + C.indices.size()) << "\n";

    for (int i = 0; i < m_nC; ++i) {
      const int stride = C.offsets[i + 1] - C.offsets[i];
      m_out << stride;
      for (int j = 0; j < stride; ++j)
        m_out << ' ' << C.indices[C.offsets[i] + j];
      m_out << '\n';
    }
  }

  template <typename DerivedC>
  void writeCells(const VtkCellType cellType,
                  const Eigen::MatrixBase<DerivedC> &C) {

    m_nC = C.rows();

    const int stride = vtkCellNumIndices[static_cast<int>(cellType)];
    if (stride == -1)
      throw std::runtime_error("[VtkIO] Unsupported cell type");

    m_out << "CELLS " << C.rows() << " " << (stride + 1) * C.rows() << "\n";
    for (int i = 0; i < C.rows(); ++i) {
      m_out << stride;
      for (int j = 0; j < stride; ++j)
        m_out << ' ' << C(i, j);
      m_out << '\n';
    }
  }

  void writeCellTypes(const std::vector<VtkCellType> &CT) {
    const int nC = CT.size();
    m_out << "CELL_TYPES " << nC << '\n';
    for (int i = 0; i < nC; ++i)
      m_out << static_cast<int>(CT[i]) << '\n';
  }

  template <VtkCellType CellType> void writeCellTypes(const int nC) {
    m_out << "CELL_TYPES " << nC << '\n';
    for (int i = 0; i < nC; ++i)
      m_out << static_cast<int>(CellType) << '\n';
  }

  void writeCellTypes(const VtkCellType cellType, const int nC) {
    m_out << "CELL_TYPES " << nC << '\n';
    for (int i = 0; i < nC; ++i)
      m_out << static_cast<int>(cellType) << '\n';
  }

  template <typename Derived>
  void writeDataArray(const std::string_view &name,
                      const Eigen::MatrixBase<Derived> &data) {
    static_assert(Derived::ColsAtCompileTime == -1 ||
                      Derived::ColsAtCompileTime == 1 ||
                      Derived::ColsAtCompileTime == 3,
                  "[VtkIO] Invalid data format");

    bool isScalar;
    if (data.cols() == 1)
      isScalar = true;
    else if (data.cols() == 3)
      isScalar = false;
    else
      throw std::runtime_error("[VtkIO] Invalid data format");

    static_assert(std::is_integral_v<typename Derived::Scalar> ||
                      std::is_floating_point_v<typename Derived::Scalar>,
                  "[VtkIO] Invalid data type");

    const std::string_view dataType =
        std::is_integral_v<typename Derived::Scalar> ? "int" : "double";

    if (isScalar) {
      m_out << "SCALARS " << name << " " << dataType << '\n';
      m_out << "LOOKUP_TABLE default\n";
    } else {
      m_out << "VECTORS " << name << " " << dataType << '\n';
    }

    m_out << data.format(Eigen::IOFormat(
        m_fullPrecision ? Eigen::FullPrecision : Eigen::StreamPrecision,
        Eigen::DontAlignCols, " ", "\n", "", "", "", "\n"));
  }

private:
  std::ofstream m_fout;
  std::ostream &m_out;

  std::string m_name = "mesh";

  bool m_fullPrecision = true;

  int m_nV = -1;
  int m_nC = -1;
  bool m_hasWrittenPointData = false;
  bool m_hasWrittenCellData = false;
};

#endif
