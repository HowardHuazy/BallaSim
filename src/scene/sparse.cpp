// Contains C++ translations of SciPy (sparsetools csr_tocsc, csr_matmat, coo_tocsr, csr_binop, csr_sum_duplicates), under their BSD licences:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/sparse.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace balla::scene::sparse {

namespace {

using Index = int64_t;

// sparsetools csr_tocsc: a counting sort by minor index; entries keep their major order.
void compress_swap(Index n_major, Index n_minor, const std::vector<Index>& ap, const std::vector<Index>& aj,
                   const std::vector<double>& ax, std::vector<Index>& bp, std::vector<Index>& bi,
                   std::vector<double>& bx) {
  const auto nnz = static_cast<Index>(ax.size());
  bp.assign(static_cast<size_t>(n_minor + 1), 0);
  bi.resize(static_cast<size_t>(nnz));
  bx.resize(static_cast<size_t>(nnz));
  for (Index n = 0; n < nnz; ++n) bp[static_cast<size_t>(aj[static_cast<size_t>(n)])]++;
  for (Index col = 0, cumsum = 0; col < n_minor; ++col) {
    const Index temp = bp[static_cast<size_t>(col)];
    bp[static_cast<size_t>(col)] = cumsum;
    cumsum += temp;
  }
  bp[static_cast<size_t>(n_minor)] = nnz;
  for (Index row = 0; row < n_major; ++row) {
    for (Index jj = ap[static_cast<size_t>(row)]; jj < ap[static_cast<size_t>(row + 1)]; ++jj) {
      const Index col = aj[static_cast<size_t>(jj)];
      const Index dest = bp[static_cast<size_t>(col)];
      bi[static_cast<size_t>(dest)] = row;
      bx[static_cast<size_t>(dest)] = ax[static_cast<size_t>(jj)];
      bp[static_cast<size_t>(col)]++;
    }
  }
  for (Index col = 0, last = 0; col <= n_minor; ++col) {
    const Index temp = bp[static_cast<size_t>(col)];
    bp[static_cast<size_t>(col)] = last;
    last = temp;
  }
}

bool sorted_indices(const Matrix& a) {
  for (Index i = 0; i < a.major(); ++i) {
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj + 1 < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      if (a.indices[static_cast<size_t>(jj)] > a.indices[static_cast<size_t>(jj + 1)]) return false;
    }
  }
  return true;
}

// csr_sort_indices: std::sort of (index, value) pairs by index per row (SciPy's kv_pair_less).
void sort_indices(Matrix& a) {
  if (sorted_indices(a)) return;
  std::vector<std::pair<Index, double>> temp;
  for (Index i = 0; i < a.major(); ++i) {
    const Index start = a.indptr[static_cast<size_t>(i)], end = a.indptr[static_cast<size_t>(i + 1)];
    temp.resize(static_cast<size_t>(end - start));
    for (Index jj = start, n = 0; jj < end; ++jj, ++n) {
      temp[static_cast<size_t>(n)] = {a.indices[static_cast<size_t>(jj)], a.data[static_cast<size_t>(jj)]};
    }
    std::sort(temp.begin(), temp.end(),
              [](const std::pair<Index, double>& x, const std::pair<Index, double>& y) { return x.first < y.first; });
    for (Index jj = start, n = 0; jj < end; ++jj, ++n) {
      a.indices[static_cast<size_t>(jj)] = temp[static_cast<size_t>(n)].first;
      a.data[static_cast<size_t>(jj)] = temp[static_cast<size_t>(n)].second;
    }
  }
}

// sparsetools csr_matmat (SMMP).
void matmat(Index n_row, Index n_col, const std::vector<Index>& ap, const std::vector<Index>& aj,
            const std::vector<double>& ax, const std::vector<Index>& bp, const std::vector<Index>& bj,
            const std::vector<double>& bx, std::vector<Index>& cp, std::vector<Index>& cj, std::vector<double>& cx) {
  std::vector<Index> next(static_cast<size_t>(n_col), -1);
  std::vector<double> sums(static_cast<size_t>(n_col), 0.0);
  cp.assign(static_cast<size_t>(n_row + 1), 0);
  cj.clear();
  cx.clear();
  Index nnz = 0;
  for (Index i = 0; i < n_row; ++i) {
    Index head = -2, length = 0;
    for (Index jj = ap[static_cast<size_t>(i)]; jj < ap[static_cast<size_t>(i + 1)]; ++jj) {
      const Index j = aj[static_cast<size_t>(jj)];
      const double v = ax[static_cast<size_t>(jj)];
      for (Index kk = bp[static_cast<size_t>(j)]; kk < bp[static_cast<size_t>(j + 1)]; ++kk) {
        const Index k = bj[static_cast<size_t>(kk)];
        sums[static_cast<size_t>(k)] += v * bx[static_cast<size_t>(kk)];
        if (next[static_cast<size_t>(k)] == -1) {
          next[static_cast<size_t>(k)] = head;
          head = k;
          length++;
        }
      }
    }
    for (Index jj = 0; jj < length; ++jj) {
      if (sums[static_cast<size_t>(head)] != 0) {
        cj.push_back(head);
        cx.push_back(sums[static_cast<size_t>(head)]);
        nnz++;
      }
      const Index temp = head;
      head = next[static_cast<size_t>(head)];
      next[static_cast<size_t>(temp)] = -1;
      sums[static_cast<size_t>(temp)] = 0.0;
    }
    cp[static_cast<size_t>(i + 1)] = nnz;
  }
}

Matrix convert(const Matrix& a, Format target) {
  if (a.format == target) return a;
  Matrix b;
  b.format = target;
  b.rows = a.rows;
  b.cols = a.cols;
  compress_swap(a.major(), a.minor(), a.indptr, a.indices, a.data, b.indptr, b.indices, b.data);
  return b;
}

}  // namespace

Matrix csr(int64_t rows, int64_t cols, std::vector<int64_t> indptr, std::vector<int64_t> indices,
           std::vector<double> data) {
  if (rows < 0 || cols < 0) throw std::invalid_argument("sparse: negative shape");
  if (static_cast<int64_t>(indptr.size()) != rows + 1 || indices.size() != data.size()) {
    throw std::invalid_argument("sparse: inconsistent CSR arrays");
  }
  // SciPy's check_format(full_check=True): indptr starts at 0, never decreases and ends at nnz; indices lie in [0, cols).
  if (indptr[0] != 0) throw std::invalid_argument("index pointer should start with 0");
  for (size_t i = 0; i + 1 < indptr.size(); ++i) {
    if (indptr[i + 1] < indptr[i]) throw std::invalid_argument("index pointer values must form a non-decreasing sequence");
  }
  if (indptr.back() != static_cast<int64_t>(indices.size())) {
    throw std::invalid_argument("last value of index pointer should be equal to the size of index and data arrays");
  }
  for (const int64_t j : indices) {
    if (j < 0 || j >= cols) throw std::invalid_argument("column index out of bounds");
  }
  return {Format::kCsr, rows, cols, std::move(indptr), std::move(indices), std::move(data)};
}

bool has_canonical_format(const Matrix& a) {
  for (Index i = 0; i < a.major(); ++i) {
    if (a.indptr[static_cast<size_t>(i)] > a.indptr[static_cast<size_t>(i + 1)]) return false;
    for (Index jj = a.indptr[static_cast<size_t>(i)] + 1; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      if (!(a.indices[static_cast<size_t>(jj - 1)] < a.indices[static_cast<size_t>(jj)])) return false;
    }
  }
  return true;
}

void sum_duplicates(Matrix& a) {
  if (has_canonical_format(a)) return;
  sort_indices(a);
  Index nnz = 0, row_end = 0;
  for (Index i = 0; i < a.major(); ++i) {
    Index jj = row_end;
    row_end = a.indptr[static_cast<size_t>(i + 1)];
    while (jj < row_end) {
      const Index j = a.indices[static_cast<size_t>(jj)];
      double x = a.data[static_cast<size_t>(jj)];
      jj++;
      while (jj < row_end && a.indices[static_cast<size_t>(jj)] == j) {
        x += a.data[static_cast<size_t>(jj)];
        jj++;
      }
      a.indices[static_cast<size_t>(nnz)] = j;
      a.data[static_cast<size_t>(nnz)] = x;
      nnz++;
    }
    a.indptr[static_cast<size_t>(i + 1)] = nnz;
  }
  a.indices.resize(static_cast<size_t>(nnz));
  a.data.resize(static_cast<size_t>(nnz));
}

Matrix coo_to_csr(int64_t rows, int64_t cols, std::span<const int64_t> row, std::span<const int64_t> col,
                  std::span<const double> data) {
  const auto nnz = static_cast<Index>(data.size());
  if (static_cast<Index>(row.size()) != nnz || static_cast<Index>(col.size()) != nnz) {
    throw std::invalid_argument("sparse: COO arrays of different lengths");
  }
  Matrix b{Format::kCsr, rows, cols, {}, {}, {}};
  if (nnz == 0) {
    b.indptr.assign(static_cast<size_t>(rows + 1), 0);
    return b;
  }
  b.indptr.assign(static_cast<size_t>(rows + 1), 0);
  b.indices.resize(static_cast<size_t>(nnz));
  b.data.resize(static_cast<size_t>(nnz));
  for (Index n = 0; n < nnz; ++n) {
    if (row[static_cast<size_t>(n)] < 0 || row[static_cast<size_t>(n)] >= rows || col[static_cast<size_t>(n)] < 0 ||
        col[static_cast<size_t>(n)] >= cols) {
      throw std::invalid_argument("sparse: COO index out of range");
    }
    b.indptr[static_cast<size_t>(row[static_cast<size_t>(n)])]++;
  }
  for (Index i = 0, cumsum = 0; i < rows; ++i) {
    const Index temp = b.indptr[static_cast<size_t>(i)];
    b.indptr[static_cast<size_t>(i)] = cumsum;
    cumsum += temp;
  }
  b.indptr[static_cast<size_t>(rows)] = nnz;
  for (Index n = 0; n < nnz; ++n) {
    const Index r = row[static_cast<size_t>(n)];
    const Index dest = b.indptr[static_cast<size_t>(r)];
    b.indices[static_cast<size_t>(dest)] = col[static_cast<size_t>(n)];
    b.data[static_cast<size_t>(dest)] = data[static_cast<size_t>(n)];
    b.indptr[static_cast<size_t>(r)]++;
  }
  for (Index i = 0, last = 0; i <= rows; ++i) {
    const Index temp = b.indptr[static_cast<size_t>(i)];
    b.indptr[static_cast<size_t>(i)] = last;
    last = temp;
  }
  sum_duplicates(b);
  return b;
}

Matrix transpose(const Matrix& a) {
  Matrix t = a;
  t.format = a.format == Format::kCsr ? Format::kCsc : Format::kCsr;
  t.rows = a.cols;
  t.cols = a.rows;
  return t;
}

Matrix to_csr(const Matrix& a) { return convert(a, Format::kCsr); }
Matrix to_csc(const Matrix& a) { return convert(a, Format::kCsc); }

Matrix matmul(const Matrix& a, const Matrix& b) {
  if (a.cols != b.rows) throw std::invalid_argument("sparse matmul: dimension mismatch");
  const Matrix o = convert(b, a.format);
  Matrix c;
  c.format = a.format;
  c.rows = a.rows;
  c.cols = b.cols;
  if (a.format == Format::kCsr) {
    matmat(a.rows, b.cols, a.indptr, a.indices, a.data, o.indptr, o.indices, o.data, c.indptr, c.indices, c.data);
  } else {
    matmat(b.cols, a.rows, o.indptr, o.indices, o.data, a.indptr, a.indices, a.data, c.indptr, c.indices, c.data);
  }
  return c;
}

std::vector<double> matvec(const Matrix& a, std::span<const double> x) {
  if (static_cast<int64_t>(x.size()) != a.cols) throw std::invalid_argument("sparse matvec: dimension mismatch");
  std::vector<double> y(static_cast<size_t>(a.rows), 0.0);
  if (a.format == Format::kCsr) {
    for (Index i = 0; i < a.rows; ++i) {
      double sum = y[static_cast<size_t>(i)];
      for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
        sum += a.data[static_cast<size_t>(jj)] * x[static_cast<size_t>(a.indices[static_cast<size_t>(jj)])];
      }
      y[static_cast<size_t>(i)] = sum;
    }
  } else {
    for (Index j = 0; j < a.cols; ++j) {
      for (Index ii = a.indptr[static_cast<size_t>(j)]; ii < a.indptr[static_cast<size_t>(j + 1)]; ++ii) {
        const auto i = static_cast<size_t>(a.indices[static_cast<size_t>(ii)]);
        y[i] += a.data[static_cast<size_t>(ii)] * x[static_cast<size_t>(j)];
      }
    }
  }
  return y;
}

std::vector<double> matvecs(const Matrix& a, std::span<const double> x, int64_t k) {
  if (static_cast<int64_t>(x.size()) != a.cols * k) throw std::invalid_argument("sparse matvecs: dimension mismatch");
  std::vector<double> y(static_cast<size_t>(a.rows * k), 0.0);
  auto axpy = [k](double s, const double* xs, double* ys) {
    for (int64_t v = 0; v < k; ++v) ys[v] += s * xs[v];
  };
  if (a.format == Format::kCsr) {
    for (Index i = 0; i < a.rows; ++i) {
      for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
        axpy(a.data[static_cast<size_t>(jj)], x.data() + a.indices[static_cast<size_t>(jj)] * k, y.data() + i * k);
      }
    }
  } else {
    for (Index j = 0; j < a.cols; ++j) {
      for (Index ii = a.indptr[static_cast<size_t>(j)]; ii < a.indptr[static_cast<size_t>(j + 1)]; ++ii) {
        axpy(a.data[static_cast<size_t>(ii)], x.data() + j * k, y.data() + a.indices[static_cast<size_t>(ii)] * k);
      }
    }
  }
  return y;
}

Matrix add(const Matrix& a_in, const Matrix& b_in) {
  if (a_in.rows != b_in.rows || a_in.cols != b_in.cols) throw std::invalid_argument("sparse add: shape mismatch");
  const Matrix a = to_csr(a_in);
  const Matrix b = to_csr(b_in);
  Matrix c{Format::kCsr, a.rows, a.cols, std::vector<Index>(static_cast<size_t>(a.rows + 1), 0), {}, {}};
  if (has_canonical_format(a) && has_canonical_format(b)) {
    for (Index i = 0; i < a.rows; ++i) {
      Index ap = a.indptr[static_cast<size_t>(i)], ae = a.indptr[static_cast<size_t>(i + 1)];
      Index bp = b.indptr[static_cast<size_t>(i)], be = b.indptr[static_cast<size_t>(i + 1)];
      auto put = [&c](Index j, double v) {
        if (v != 0) {
          c.indices.push_back(j);
          c.data.push_back(v);
        }
      };
      while (ap < ae && bp < be) {
        const Index aj = a.indices[static_cast<size_t>(ap)], bj = b.indices[static_cast<size_t>(bp)];
        if (aj == bj) {
          put(aj, a.data[static_cast<size_t>(ap)] + b.data[static_cast<size_t>(bp)]);
          ap++;
          bp++;
        } else if (aj < bj) {
          put(aj, a.data[static_cast<size_t>(ap)] + 0.0);
          ap++;
        } else {
          put(bj, 0.0 + b.data[static_cast<size_t>(bp)]);
          bp++;
        }
      }
      for (; ap < ae; ++ap) put(a.indices[static_cast<size_t>(ap)], a.data[static_cast<size_t>(ap)] + 0.0);
      for (; bp < be; ++bp) put(b.indices[static_cast<size_t>(bp)], 0.0 + b.data[static_cast<size_t>(bp)]);
      c.indptr[static_cast<size_t>(i + 1)] = static_cast<Index>(c.data.size());
    }
    return c;
  }
  std::vector<Index> next(static_cast<size_t>(a.cols), -1);
  std::vector<double> arow(static_cast<size_t>(a.cols), 0.0), brow(static_cast<size_t>(a.cols), 0.0);
  for (Index i = 0; i < a.rows; ++i) {
    Index head = -2, length = 0;
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      const Index j = a.indices[static_cast<size_t>(jj)];
      arow[static_cast<size_t>(j)] += a.data[static_cast<size_t>(jj)];
      if (next[static_cast<size_t>(j)] == -1) {
        next[static_cast<size_t>(j)] = head;
        head = j;
        length++;
      }
    }
    for (Index jj = b.indptr[static_cast<size_t>(i)]; jj < b.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      const Index j = b.indices[static_cast<size_t>(jj)];
      brow[static_cast<size_t>(j)] += b.data[static_cast<size_t>(jj)];
      if (next[static_cast<size_t>(j)] == -1) {
        next[static_cast<size_t>(j)] = head;
        head = j;
        length++;
      }
    }
    for (Index jj = 0; jj < length; ++jj) {
      const double result = arow[static_cast<size_t>(head)] + brow[static_cast<size_t>(head)];
      if (result != 0) {
        c.indices.push_back(head);
        c.data.push_back(result);
      }
      const Index temp = head;
      head = next[static_cast<size_t>(head)];
      next[static_cast<size_t>(temp)] = -1;
      arow[static_cast<size_t>(temp)] = 0.0;
      brow[static_cast<size_t>(temp)] = 0.0;
    }
    c.indptr[static_cast<size_t>(i + 1)] = static_cast<Index>(c.data.size());
  }
  return c;
}

Matrix scaled(const Matrix& a, double s) {
  Matrix c = a;
  for (double& x : c.data) x = x * s;
  return c;
}

namespace {

// COO of a compressed matrix in its storage order (tocoo), each value times w[col] or w[row], then tocsr.
Matrix multiply_coo(const Matrix& a, std::span<const double> w, bool by_column) {
  const auto nnz = static_cast<size_t>(a.nnz());
  std::vector<Index> row(nnz), col(nnz);
  std::vector<double> data(nnz);
  for (Index i = 0; i < a.major(); ++i) {
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      const Index r = a.format == Format::kCsr ? i : a.indices[static_cast<size_t>(jj)];
      const Index c = a.format == Format::kCsr ? a.indices[static_cast<size_t>(jj)] : i;
      row[static_cast<size_t>(jj)] = r;
      col[static_cast<size_t>(jj)] = c;
      data[static_cast<size_t>(jj)] = a.data[static_cast<size_t>(jj)] * w[static_cast<size_t>(by_column ? c : r)];
    }
  }
  return coo_to_csr(a.rows, a.cols, row, col, data);
}

}  // namespace

Matrix multiply_columns(const Matrix& a, std::span<const double> w) {
  if (static_cast<int64_t>(w.size()) != a.cols) throw std::invalid_argument("sparse multiply: shape mismatch");
  return multiply_coo(a, w, true);
}

Matrix multiply_rows(const Matrix& a, std::span<const double> w) {
  if (static_cast<int64_t>(w.size()) != a.rows) throw std::invalid_argument("sparse multiply: shape mismatch");
  return multiply_coo(a, w, false);
}

std::vector<double> diagonal(const Matrix& a_in) {
  const Matrix a = to_csr(a_in);
  const Index n = std::min(a.rows, a.cols);
  std::vector<double> y(static_cast<size_t>(n));
  for (Index i = 0; i < n; ++i) {
    double diag = 0.0;
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      if (a.indices[static_cast<size_t>(jj)] == i) diag += a.data[static_cast<size_t>(jj)];
    }
    y[static_cast<size_t>(i)] = diag;
  }
  return y;
}

Matrix add_diagonal(const Matrix& a, std::span<const double> d) {
  const auto n = static_cast<Index>(d.size());
  if (a.rows != n || a.cols != n) throw std::invalid_argument("sparse add_diagonal: shape mismatch");
  Matrix dm{Format::kCsr, n, n, std::vector<Index>(static_cast<size_t>(n + 1), 0), {}, {}};
  for (Index i = 0; i < n; ++i) {
    if (d[static_cast<size_t>(i)] != 0) {
      dm.indices.push_back(i);
      dm.data.push_back(d[static_cast<size_t>(i)]);
    }
    dm.indptr[static_cast<size_t>(i + 1)] = static_cast<Index>(dm.data.size());
  }
  return add(a, dm);
}

Matrix kron_identity3(const Matrix& a_in) {
  const Matrix a = to_csr(a_in);
  const auto nnz = static_cast<size_t>(a.nnz());
  std::vector<Index> row(3 * nnz), col(3 * nnz);
  std::vector<double> data(3 * nnz);
  for (Index i = 0; i < a.rows; ++i) {
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      for (Index k = 0; k < 3; ++k) {
        const auto e = static_cast<size_t>(3 * jj + k);
        row[e] = 3 * i + k;
        col[e] = 3 * a.indices[static_cast<size_t>(jj)] + k;
        data[e] = a.data[static_cast<size_t>(jj)] * 1.0;
      }
    }
  }
  return coo_to_csr(3 * a.rows, 3 * a.cols, row, col, data);
}

int64_t bandwidth(const Matrix& a) {
  int64_t bw = 0;
  for (Index i = 0; i < a.major(); ++i) {
    for (Index jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      bw = std::max<int64_t>(bw, std::llabs(a.indices[static_cast<size_t>(jj)] - i));
    }
  }
  return bw;
}

}  // namespace balla::scene::sparse
