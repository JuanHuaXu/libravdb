# LibraVDB — Exhaustive Code Analysis

**Date:** 2026-04-27
**Version:** 1.0.0
**Go version:** 1.25+
**Repository:** `github.com/xDarkicex/libravdb`
**Non-test Go files:** 73 · **Total lines:** ~26,730

---

## 1. Architecture Overview

LibraVDB is an embedded vector database library organized into **four principal layers** + infrastructure:

```
Application API  (libravdb/ — Database, Collection, Query, Streaming, Batch, Tx)
    ↓
Index Layer  (internal/index/ — HNSW, Flat, IVF-PQ + factory & interfaces)
    ↓
Storage Layer  (internal/storage/ — singlefile engine, LSM+WAL, segments, registry)
    ↓
OS (page-based file I/O with fsync)
```

Surrounding infrastructure:
- **Quantization** (`internal/quant/`) — PQ and Scalar
- **Memory** (`internal/memory/`) — LRU cache, mmap, manager, monitor
- **Observability** (`internal/obs/`) — Prometheus metrics, circuit breaker, health
- **Utilities** (`internal/util/`) — distance functions, heaps, validation, encoding
- **Filter** (`internal/filter/`) — parser, range, equality, containment, logical ops

---

## 2. Core Package (`libravdb/`)

### 2.1 `database.go` (~541 lines)

- `Database` struct: holds `collections map[string]*Collection`, `storage storage.Engine`, `metrics`, `health`, `config`.
- Collection lifecycle: `CreateCollection()`, `DropCollection()`, `ListCollections()`, `GetCollection()`.
- Health monitoring via `obs.HealthChecker`.
- Thread-safe via `sync.RWMutex`.

### 2.2 `collection.go` (~2,326 lines)

**Heart of the library.** Each `Collection` wraps:
- `index.Index` — the vector index (HNSW/Flat/IVF-PQ)
- `storage.Collection` — persistence layer
- `memory.MemoryManager` — memory limits/eviction
- `obs.Metrics` — per-collection counters

**Key methods:**
- `Insert()` — assign ordinal → insert into index → write to storage → trigger GC if needed
- `InsertBatch()` — batch ordinal assignment → bulk insert → batch flush
- `Search()` — query index → filter processing → assemble results
- `Upsert()` — CAS-based conditional update (version check)
- `Delete()` — mark deleted in index + storage
- `StreamingInsert()` — streaming/chunked insert with progress callbacks
- `RebuildIndex()` — rebuild from storage for recovery

**Config schema (CollectionConfig):**
- `Dimension`, `Metric` (0=L2, 1=Inner, 2=Cosine)
- `IndexType` (0=HNSW, 1=IVF-PQ, 2=Flat)
- `M`, `EfConstruction`, `EfSearch`
- `NClusters`, `NProbes` (IVF-PQ)
- `ML` (level generation factor)
- `Version`

### 2.3 `query.go` (~563 lines)

- `Query` builder with fluent API: `Filter()`, `Limit()`, `Offset()`, `IncludeVectors()`, `IncludeMetadata()`, `Metric()`, `EF()`, `K()`.
- `Query.Execute()` on a collection returns `*SearchResults`.

### 2.4 `options.go` (~387 lines)

- `DatabaseConfig`, `CollectionConfig` option functions via functional options pattern.
- Default collection config: 3 dims, Cosine, HNSW, M=32, EF=200, ML=1/ln(2).

### 2.5 `types.go` (~290 lines)

- `DistanceMetric`, `IndexType` enums.
- `VectorEntry` — the core data carrier (ID, Ordinal, Vector, Metadata, Version).
- `SearchResult`, `SearchResults`.

### 2.6 `streaming.go` (~1,448 lines)

- `StreamingWriter` — buffers inserts and flushes in configurable batches.
- `StreamingBatchInsert()` — chunked insert with progress callbacks.
- Backpressure via context cancellation.

### 2.7 `tx.go` (~685 lines)

- Transactional multi-collection operations.
- `BeginTransaction()`, `Commit()`, `Rollback()`.
- CAS-based conditional writes with version conflict detection.

### 2.8 `batch.go` (~1,427 lines)

- Batch insert with bulk ordinal assignment.
- `BatchInsert()` — assigns ordinals in bulk → inserts all → flushes once.
- `BatchUpsert()` — conditional batch upserts.
- `BatchDelete()` — bulk deletion.
- `BatchSearch()` — bulk search.

### 2.9 `degradation.go` (~525 lines)

- Auto-degradation: if quantization training fails, falls back to uncompressed.
- Degradation thresholds and recovery.

### 2.10 `health_monitor.go` (~414 lines)

- `HealthMonitor` struct monitoring database health.
- Periodic health checks: collection count, storage size, memory pressure, index health.
- `CheckHealth()` returns `*HealthStatus`.

### 2.11 `recovery.go` (~205 lines)

- `RecoverFromWAL()` — recovery from WAL entries.
- `RebuildFromStorage()` — rebuild index from persisted data.
- `ValidateIntegrity()` — CRC checks, ordinal gap detection.

### 2.12 `worker_pool.go` (~95 lines)

- `WorkerPool` — fixed-size goroutine pool for parallel operations.

### 2.13 `write_controller.go` (~92 lines)

- `WriteController` — throttles writes based on memory pressure.

### 2.14 `score.go` (~37 lines)

- Score normalization helpers.

### 2.15 `collection_sharding.go` (~188 lines)

- `ShardConfig` and sharding utilities for partitioning collections.

### 2.16 `batch_errors.go` — Batch-specific error types and handling.

### 2.17 `errors.go` — Core error types (ErrCollectionNotFound, ErrInvalidDimension, etc.).

---

## 3. Index Layer

### 3.1 Interface (`internal/index/interfaces.go`)

```go
type Index interface {
    Insert(ctx context.Context, entry *VectorEntry) error
    BatchInsert(ctx context.Context, entries []*VectorEntry) error
    Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error)
    Delete(ctx context.Context, id string) error
    Size() int
    MemoryUsage() int64
    Close() error
    SaveToDisk(ctx context.Context, path string) error
    LoadFromDisk(ctx context.Context, path string) error
    GetPersistenceMetadata() *PersistenceMetadata
}
```

Three implementations: **HNSW**, **Flat**, **IVF-PQ**, each wrapped by an adapter (hnswWrapper, ivfpqWrapper, flatWrapper) to convert between interface types and internal types.

### 3.2 HNSW (`internal/index/hnsw/` — ~8 Go files)

**Core (`hnsw.go`, ~1,100+ lines):**
- `Index` struct with multi-layer graph: `nodes []*Node`, `entryPoint *Node`, `levelGenerator`, `distance` func.
- Each node: `Ordinal`, `Level`, `Links [][]uint32`, `CompressedVector []byte`.
- Insert: level generation via `level = floor(-ln(unif) * ml)` (capped at 16 levels).
- Greedy search: top-down from maxLevel → level 0.
- Batch insert: chunks of 100 for large batches.
- Supports **quantization** (trains on first N vectors, then compresses).
- Supports **raw vector stores** (InMemory or Slabby via `RawVectorStore` interface).
- Memory mapping support: `EnableMemoryMapping()`, `DisableMemoryMapping()`, `IsMemoryMapped()`.
- Entry point candidates: nodes with level ≥ 2.
- `RawVectorStoreProfile()` exposes backend stats (Slabby: segment capacity, utilization).

**Node (`node.go`):** Graph node with links and optional compressed vector.

**Config (`params.go`):** HNSWConfig: Dimension, M, EF, ML, Metric, RandomSeed, Provider, RawVectorStore, Quantization.

**Insert (`insert.go`):** `insertNode()` handles neighbor selection at each layer, pruning to M neighbors using greedy search.

**Search (`search.go`):** `searchScratch` struct, `candidateMinHeap`, `candidateMaxHeap` (priority queues), siftUp/siftDown logic. Phase 1: top-down greedy search. Phase 2: level 0 with EF candidate list. Supports Epsilon search.

**Delete (`delete.go`):** Lazy deletion — marks node as deleted but doesn't remove from graph. Graph pruning follows to maintain quality.

**Neighbors (`neighbors.go`):** `NeighborSelector` with max-degree pruning, candidate filtering.

**Binary format (`format.go`):** `IndexFileHeader` (128 bytes, magic "HNSWVIDX"), `NodeEntry` (variable-length: ID + vector + metadata), `LinkEntry`, `ConfigEntry`, `MetadataEntry`. File layout: header → config → nodes → links → metadata. CRC32 integrity.

**Persistence (`persistence.go`, ~656 lines):** `saveToDiskImpl()` / `loadFromDiskImpl()` — atomic write via temp file + rename. Nodes saved in chunks (1000). Links saved per node with level metadata. Entry point preserved. `rebuildIndexState()` after load.

**Vector stores:**
- `InMemoryRawVectorStore` (149 lines): `[][]float32` with slot references, tracking active count and bytes.
- `SlabbyRawVectorStore` (212 lines): Uses `slabby.SlotArena` for slab allocation with PCPU cache. Segment-based (default 4096 slots/segment). Auto-grows. Profile exposes reserved/live/free bytes and utilization.

### 3.3 Flat Index (`internal/index/flat/flat.go`)

- Brute-force exact search: stores all vectors in `[][]float32`.
- `Config`: Dimension, Metric, Quantization.
- `PersistenceMetadata` (version, node_count, dimension, size, checksum).
- `Index` with `sync.RWMutex`.
- `BatchInsert` — bulk vector copy.

### 3.4 IVF-PQ Index (`internal/index/ivfpq/ivfpq.go`)

- **Config:** Dimension, NClusters (default sqrt(N)), NProbes (default 25% probe ratio), Metric, Quantization, MaxIterations (100), Tolerance (1e-6), RandomSeed.
- Auto-tuning available.
- k-means clustering for partitioning + PQ for compression.
- Search: find best NProbes clusters → PQ distance computation.

### 3.5 Index Factory (`internal/index/registry.go`)

- `IndexFactory` creates HNSW/Flat/IVF-PQ by IndexType.
- `DefaultIndexFactory` global instance.

---

## 4. Quantization (`internal/quant/`)

### 4.1 Interface (`interfaces.go`)

```go
type Quantizer interface {
    Train(ctx context.Context, vectors [][]float32) error
    Compress(vector []float32) ([]byte, error)
    Decompress(data []byte) ([]float32, error)
    Distance(compressed1, compressed2 []byte) (float32, error)
    DistanceToQuery(compressed []byte, query []float32) (float32, error)
    CompressionRatio() float32
    MemoryUsage() int64
    IsTrained() bool
    Config() *QuantizationConfig
}
```

### 4.2 Config (`QuantizationConfig`)

- `Type` (Product or Scalar)
- `Bits` (1-32, default 8)
- `Codebooks` (for PQ, default 8)
- `TrainRatio` (0.0-1.0, default 0.1)
- `CacheSize` (for PQ, default 1000)

### 4.3 Product Quantizer (`product.go`)

- K-means clustering on subspaces.
- `centroids [][][]float32` — [subspace][centroid][dimension]
- `distanceTables [][]float32` — precomputed for fast query
- Compress packs codes into bytes via bit-packing.
- Training: k-means with configurable iterations/tolerance.
- Default: 8 codebooks, 8 bits → 16:1 compression for 128-dim vectors.

### 4.4 Scalar Quantizer (`scalar.go`)

- Linear quantization per dimension.
- Tracks min/max/scale/offset per dimension.
- Default: 8 bits → 4:1 compression.
- Simpler but lower quality than PQ.

### 4.5 Registry (`registry.go`)

- Global registry with `Register()`/`Create()`.
- Auto-registers PQ and Scalar factories on init.

### 4.6 Error System (`errors.go`)

- `QuantizationRecoveryManager` with retry strategies: reduce codebooks/bits, reduce training ratio, fallback to uncompressed.
- `ValidateQuantizationHealth()` — checks trained state and config validity.

---

## 5. Storage Layer

### 5.1 Engine Interface (`internal/storage/interfaces.go`)

```go
type Engine interface {
    CreateCollection(name string, config interface{}) (Collection, error)
    GetCollection(name string) (Collection, error)
    ListCollections() ([]string, error)
    DeleteCollection(name string) error
    Close() error
}

type Collection interface {
    AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error
    Insert(ctx context.Context, entry *index.VectorEntry) error
    InsertBatch(ctx context.Context, entries []*index.VectorEntry) error
    Exists(ctx context.Context, id string) (bool, error)
    Get(ctx context.Context, id string) (*index.VectorEntry, error)
    GetIDByOrdinal(ctx context.Context, ordinal uint32) (string, error)
    MemoryUsage(ctx context.Context) (int64, error)
    Delete(ctx context.Context, id string) error
    Iterate(ctx context.Context, fn func(*index.VectorEntry) error) error
    Count(ctx context.Context) (int, error)
    Close() error
}

type TransactionalEngine interface {
    PrepareTx(ctx context.Context, ops []TxOperation) ([]TxOperation, error)
    CommitTx(ctx context.Context, ops []TxOperation) error
}
```

### 5.2 Single-File Engine (`internal/storage/singlefile/` — ~3,500+ lines)

**The primary storage engine.** A single `.libravdb` file with:

**File Layout (1.0 spec):**
- **Page 0:** File header (magic "LIBRAVDB", 0x4C564442, format version 1, page size 4096, file ID, timestamp, active metapage, WAL pointers, CRC32 Castagnoli).
- **Pages 1-2:** Dual metapages (A/B with alternating epoch) for crash recovery.
  - `metaPage`: Magic (0x4C56444D), epoch, root catalog, freelist, LSN, page count, collection count, snapshot offset/length, CRC32.
- **Page 3+:** WAL region with chunk-based records.
- **Data:** After WAL grows.

**WAL Protocol:**
- `chunkTypeSnapshot` (1) = full state checkpoint
- `chunkTypeWAL` (2) = incremental transaction frames
- Record types: TxBegin(1), TxCommit(2), TxAbort(3), CollectionCreate(10), CollectionDelete(11), RecordPut(20), RecordDelete(21)
- Each frame: 40-byte `walFrameHeader` (magic, version, recordType, LSN, TxID, prevLSN, payloadLen, CRC32) + payload
- Chunks: 16-byte `chunkHeader` (magic, kind, version, payloadLen, CRC32)

**Atomic Persistence:**
- `checkpointThresholdBytes = 256MB`, `checkpointThresholdOps = 65536`
- `batchSize = 256`, `batchFlushInterval = 10ms`
- **Group commit window:** adaptive 1-5ms to coalesce concurrent flushes
- **Batch flusher goroutine:** background ticker (10ms) + event-driven wakeup via channel

**Snapshot encoding:** Custom binary codec with pooled `binaryEncoder` (slab allocator, 256B-16KB capacity). Stores entire `persistedState` (collections → records) with typed metadata (bool, string, int, int64, float32, float64, []string, []interface{}, map).

**Recovery:** On open, reads metapages (chooses higher epoch), loads snapshot, replays committed WAL transactions. Discards uncommitted. Tracks replay/discard stats.

**Transactions:** Multi-collection atomic writes. Each operation gets LSN. Frames grouped by TxID. `PrepareTx()` validates + assigns ordinals; `CommitTx()` writes all frames atomically.

### 5.3 LSM Engine (`internal/storage/lsm/`)

- `Engine` — wrapper managing collections with per-collection WAL.
- `Collection` — in-memory cache (`map[string]*VectorEntry`) + `ordinalToID []string` + per-collection WAL file.
- `recoverFromWAL()` — rebuilds cache from WAL on startup.
- Simpler than singlefile; uses separate WAL files per collection.

### 5.4 WAL (`internal/storage/wal/`)

- Append-only file with length-prefixed entries.
- Entry format: timestamp(8) + operation(1) + ID length(4) + ID + vector length(4) + vector(float32[]) + metadata length(4) + metadata(JSON).
- `Append()` / `AppendBatch()` — buffered writes + file.Sync() for durability.
- `Read()` — replay for recovery.

### 5.5 Segments (`internal/storage/segments/`)

- Segment format with header (magic, version, compression, offset, checksum).
- Chunk-based storage with reader/writer.

---

## 6. Memory Management (`internal/memory/`)

### 6.1 LRU Cache (`cache.go`)

- Thread-safe `LRUCache` backed by `container/list`.
- Byte-sized eviction, capacity in bytes.
- `Evict(bytes)` frees specified bytes from LRU end.

### 6.2 Memory Manager (`manager.go`)

- **Memory tiers:** Hot (RAM), Warm (compressed), Cold (mmap).
- Tracks: heap usage via `runtime.MemStats`, cache usage, mmap usage.
- **Pressure levels:** Low(70%), Moderate(80%), High(90%), Critical(95%).
- **Pressure response:** Evict caches → enable mmap → trigger GC.
- **Config:** MaxMemory, PressureThresholds, MonitorInterval(5s), EnableGC, GCThreshold(85%), EnableMMap, MMapThreshold(100MB), MMapPath.

### 6.3 Memory Manager Recovery (`errors.go`)

- `MemoryRecoveryManager` — 3-phase recovery: lightweight GC → moderate recovery → aggressive GC.
- `MemoryHealthMonitor` — 80%/90%/95% warning thresholds.

### 6.4 Memory Mapping (`mmap.go` / `mmap_windows.go` / `mmap_unsupported.go`)

- Unix: `unix.Mmap`/`unix.Munmap`/`unix.Msync` with PROT_READ/PROT_WRITE.
- Windows: `windows.CreateFileMapping`/`MapViewOfFile`/`FlushViewOfFile`.
- Resizable, syncable, closeable. `MemoryMapManager` manages multiple mappings.

### 6.5 Monitor (`monitor.go`)

- `MemorySnapshot` with HeapAlloc/Sys/Inuse/Released, GC stats.
- `CalculateMemoryTrend()` — >1MB/s = increasing, <-1MB/s = decreasing.
- `ForceGC()` — returns freed bytes.

### 6.6 Interfaces (`interfaces.go`)

- `MemoryManager` interface with RegisterCache, RegisterMemoryMappable, Evict, SetLimit.
- `Cache` interface with Evict, Size, Clear, Name.
- `MemoryMappable` interface with CanMemoryMap, EnableMemoryMapping.

---

## 7. Observability (`internal/obs/`)

### 7.1 Metrics (`metrics.go`)

**Prometheus metrics via promauto:**
| Counter | Name |
|---------|------|
| VectorInserts | libravdb_vector_inserts_total |
| VectorUpdates | libravdb_vector_updates_total |
| VectorDeletes | libravdb_vector_deletes_total |
| TxBegins | libravdb_transactions_begun_total |
| TxCommits | libravdb_transactions_committed_total |
| TxRollbacks | libravdb_transactions_rolled_back_total |
| TxConflicts | libravdb_transaction_conflicts_total |
| CASSuccesses | libravdb_cas_success_total |
| CASConflicts | libravdb_cas_conflict_total |
| CASAborts | libravdb_cas_abort_total |
| SearchQueries | libravdb_search_queries_total |
| SearchErrors | libravdb_search_errors_total |

| Histogram | Name | Buckets |
|-----------|------|---------|
| TxCommitOps | libravdb_transaction_commit_ops | 1,2,4,8,16,32,64,128,256 |
| TxCommitLatency | libravdb_transaction_commit_latency_seconds | DefBuckets |
| SearchLatency | libravdb_search_latency_seconds | DefBuckets |

Singleton pattern with `ResetForTesting()` for test isolation.

### 7.2 Circuit Breaker (`circuit.go`)

- Three states: CLOSED → OPEN → HALF_OPEN → CLOSED.
- Opens after 5 failures or 60% failure rate (min 10 requests).
- Half-open tests with 3 max requests, 30s timeout.
- `CircuitBreakerManager` — multi-breaker registry.

### 7.3 Health (`health.go`)

- `HealthChecker` with simple status checks.
- Returns `HealthStatus` with per-check results.

### 7.4 Tracing (`tracing.go`)

- Currently empty — tracing hooks not yet implemented.

---

## 8. Filter Package (`internal/filter/`)

- **`parser.go`:** `FilterParser` with schema validation, generates typed filter expressions.
- **`logical.go`:** AND/OR/NOT composition.
- **`range.go`:** GTE/LTE/GT/LT comparisons.
- **`equality.go`:** EQ/NEQ with FieldType-aware comparison.
- **`containment.go`:** IN/NOT_IN/EXISTS for array fields.
- **Type system:** FieldType enum with type-safe comparison.

---

## 9. Utilities (`internal/util/`)

### 9.1 Distance (`distance.go`)

- **L2Distance:** sqrt(sum of squared differences)
- **InnerProduct:** negated dot product (negative for max-heap behavior)
- **CosineDistance:** 1 - (dot / (|a|·|b|))

### 9.2 Heap (`heap.go`)

- `MinHeap` and `MaxHeap` implementing `container/heap` interface.
- `Candidate` struct with ID and Distance.

### 9.3 Encoding + Validation (`encoding.go`, `validation.go`)

- Vector validation: dimension check, NaN/Inf detection.

---

## 10. Strengths

1. **Mature storage engine** — Single-file engine with dual-metapage crash recovery, group commit, batch WAL, adaptive flush windows, and custom binary codec. This is production-grade.
2. **Three index algorithms** — HNSW for ANN, Flat for exact, IVF-PQ for memory-constrained high-dim.
3. **Quantization support** — PQ (16:1 for 128-dim) and Scalar (4:1) with auto-training and recovery strategies.
4. **Memory management** — Multi-tier (hot/warm/cold) with LRU eviction, OS-level mmap, pressure-driven GC.
5. **Observability** — 12+ Prometheus counters + 3 histograms, circuit breaker pattern, health monitoring.
6. **Transactional** — Multi-collection atomic writes with CAS version conflict detection.
7. **Binary protocol** — CRC32 Castagnoli everywhere (header, metapage, chunk, WAL frame, codec payloads).
8. **Clean architecture** — Well-separated layers with interfaces, factory pattern for indexes.

## 11. Risks & Concerns

1. **Graveyard of dead code / stubs** — `internal/storage/segments/` (all files are empty `package segments`), `internal/storage/registry.go` (empty), `internal/obs/tracing.go` (empty). These suggest incomplete features or abandoned plans.
2. **HNSW binary format inconsistency** — `format.go` defines `IndexFileHeader` (magic "HNSWVIDX") but `persistence.go` writes a different format (magic 0x484E5357 = "HNSW"). The file layouts described in format.go (config section, nodes section, links section, metadata section) don't match the actual write/read code which uses a simpler header(4+4+8+4) + config + nodes + links + metadata layout.
3. **HNSW persistence path is legacy** — The `rawVectorStore` (InMemory/Slabby) is the primary storage path. The HNSW binary format (persistence.go) appears to be a legacy/self-contained path that's not integrated with the main storage engine. Vectors are stored separately in the raw vector store, not in the binary index file.
4. **Lock contention** — Both `Database` and `Collection` use coarse `sync.RWMutex`. For write-heavy workloads, the single mutex on `Collection` becomes a bottleneck.
5. **Memory-mapped HNSW is half-baked** — `EnableMemoryMapping()` saves to disk then clears in-memory data, but the `mmap` implementation in `memory/mmap.go` uses `unix.Mmap` which requires explicit `Munmap`/`Mmap` cycle for resizing. The HNSW mmap path doesn't actually use the mmap package — it just saves and clears.
6. **Quota of HNSW levels** — Capped at 16 levels. For 10M vectors with ML=1/ln(2), expected max level is ~15.6, so this is technically fine but the hard cap could cause issues with pathological random sequences.
7. **Empty segment files** — `catalog.go`, `header.go`, `reader.go`, `writer.go` are all empty. This is a clear code hygiene issue.
8. **Metadata encoding type coverage** — Supports nil, bool, string, int, int64, float32, float64, []string, []interface{}, map[string]interface{} but NOT uint, uint64, complex types.
9. **No index compaction or cleanup** — HNSW lazy deletion means deleted nodes accumulate in the graph indefinitely. No compaction mechanism exists.
10. **IVF-PQ persistence is incomplete** — `GetPersistenceMetadata()` returns placeholder values (ChecksumCRC32=0, FileSize=0) with TODO comments.
11. **No benchmark results in code** — Despite 8 benchmark test files, no actual benchmark data or results are embedded in the analysis.

---

## 12. Key Design Decisions

1. **Ordinal-based addressing** — Vectors indexed by `uint32` ordinal for cache-friendly array access, with ID→ordinal and ordinal→ID maps for string lookups.
2. **Group commit** — Adaptive 1-5ms window to coalesce concurrent WAL flushes into a single fsync.
3. **Dual-metapage** — Mirrors SQLite/WAL file patterns for crash-safe metadata updates.
4. **Slab allocator for vectors** — Uses `slabby.SlotArena` (external dependency) for memory-efficient raw vector storage.
5. **Binary codec pool** — `binaryEncoderPool` with exponential growth (256B → 16KB max) reduces allocation pressure.
6. **Lazy HNSW deletion** — Deleted nodes aren't removed from the graph; marked and pruned during subsequent operations.

---

## 13. Code Metrics Summary

| Component | Files | Lines | Key Types |
|-----------|-------|-------|-----------|
| libravdb/ core | 17 | ~9,223 | Database, Collection, Streaming, Tx, Batch |
| index/hnsw | 9 | ~3,800 | Index, Node, Config, VectorStore |
| index/flat | 1 | ~800 | Index, Config, VectorEntry |
| index/ivfpq | 1 | ~2,500 | Index, Config, Cluster |
| index/ | 1 | ~600 | Index interface, factory, adapters |
| storage/singlefile | 2 | ~3,500 | Engine, Collection, codec, WAL frames |
| storage/lsm | 2 | ~600 | Engine, Collection |
| storage/wal | 2 | ~400 | WAL, Entry |
| storage/segments | 4 | 0 | (empty) |
| storage/ | 1 | 0 | (empty) |
| quant | 5 | ~2,600 | ProductQuantizer, ScalarQuantizer |
| memory | 8 | ~3,000 | LRUCache, MemoryManager, MemoryMap |
| obs | 4 | ~900 | CircuitBreaker, Metrics, HealthChecker |
| util | 4 | ~200 | DistanceFunc, MinHeap, MaxHeap |
| filter | 6 | ~1,200 | FilterParser, typed expressions |
| examples | 2 | ~200 | Usage examples |
| **TOTAL** | **73** | **~26,730** | |

---

*Analysis based on reading all 73 non-test Go source files. No test files were analyzed.*
