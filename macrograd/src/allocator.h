#ifndef allocator_h
#define allocator_h
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

// bump allocator constants
#define ARENA_PAGE_SIZE (1024 * 1024 * 1024) // 1GB, benchmarking in large tensors
#define MAX_PAGES 1024

typedef struct
{
    uint8_t* pages[MAX_PAGES];
    int num_pages;
    int top;
} Arena;

// ==== ARENA FUNCTIONS ====
static inline void arena_init(Arena* g_arena)
{
    if (g_arena->num_pages > 0)
        return;
    g_arena->pages[0] = (uint8_t*) malloc(ARENA_PAGE_SIZE);
    if (!g_arena->pages[0])
    {
        fprintf(stderr, "Arena init failed\n");
        exit(1);
    }
    g_arena->num_pages = 1;
    g_arena->top = 0;
}

static inline void arena_grow(Arena* g_arena)
{
    if (g_arena->num_pages >= MAX_PAGES)
    {
        fprintf(stderr, "Arena overflow\n");
        exit(1);
    }
    g_arena->pages[g_arena->num_pages] = (uint8_t*) malloc(ARENA_PAGE_SIZE);
    if (!g_arena->pages[g_arena->num_pages])
    {
        fprintf(stderr, "Arena grow failed\n");
        exit(1);
    }
    g_arena->num_pages++;
}

// Helper to bump up to 32 byte allocation for later AVX SIMD instructions
static inline size_t _align32(size_t size)
{
    return (size + 31) & ~31;
}

static inline void* arena_alloc(Arena* g_arena, size_t size)
{
    if (g_arena->num_pages == 0)
        arena_init(g_arena);
    size_t aligned_size = _align32(size);

    if (g_arena->top + aligned_size > ARENA_PAGE_SIZE)
    {
        arena_grow(g_arena);
        g_arena->top = 0;
    }
    void* ptr = g_arena->pages[g_arena->num_pages - 1] + g_arena->top;
    g_arena->top += aligned_size;

    return ptr;
}

static inline int get_arena_top(Arena* g_arena)
{
    return g_arena->top;
}

static inline void free_arena(Arena* g_arena)
{
    for (int i = 0; i < g_arena->num_pages; i++)
    {
        free(g_arena->pages[i]);
        g_arena->pages[i] = NULL;
    }
    g_arena->num_pages = 0;
    g_arena->top = 0;
}

#endif
