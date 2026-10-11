---
description: 
alwaysApply: true
---

# Claude Code Guidelines

## Think Before Coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## Output

- Return code first. Explanation after, only if non-obvious.
- No inline prose. Use comments sparingly - only where logic is unclear.
- No boilerplate unless explicitly requested.

## Code Rules

- Simplest working solution. No over-engineering.
- No abstractions for single-use operations.
- No speculative features or "you might also want..."
- No "flexibility" or "configurability" that wasn't requested.
- Read the file before modifying it. Never edit blind.
- No docstrings or type annotations on code not being changed.
- No error handling for scenarios that cannot happen.
- Three similar lines is better than a premature abstraction.
- If you write 200 lines and it could be 50, rewrite it.

## Review Rules

- State the bug. Show the fix. Stop.
- No suggestions beyond the scope of the review.
- No compliments on the code before or after the review.

## Debugging Rules

- Never speculate about a bug without reading the relevant code first.
- State what you found, where, and the fix. One pass.
- If cause is unclear: say so. Do not guess.

## Surgical Changes

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- Every changed line should trace directly to the user's request.

## Goal-Driven Execution

- Transform tasks into verifiable goals before implementing.
- For multi-step tasks, state a brief plan with verify steps:
  1. [Step] -> verify: [check]
  2. [Step] -> verify: [check]
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

## Simple Formatting

- No em dashes, smart quotes, or decorative Unicode symbols.
- Plain hyphens and straight quotes only.
- Natural language characters (accented letters, CJK, etc.) are fine when the content requires them.
- Code output must be copy-paste safe.

## Architecture

All code must follow this strict layered dependency architecture. The full request path is:

```text
CLIENT REQUEST
      |
      v
Router (routers/)              - FastAPI endpoints, decorators, DI
      |
      v
Decorators (decorators/)       - @http_error_handler, @enforce (IAM/RBAC/rate-limit)
      |
      v
Schemas (schemas/)             - Pydantic request/response models
      |
      v
ServiceContainer (services/)   - Business logic, orchestration
      |
      v
RepositoryContainer (repos/)   - DB access via UDFs/stored procedures
      |
      v
Database Session (PostgreSQL)  - UDFs, stored procedures, constraints
```

- Dependencies flow **downward only** -- upper layers depend on lower layers, never the reverse.
- Each layer is injected via interfaces (Inversion of Control).

### Layer Responsibilities

- **Router**: endpoint definition, `@http_error_handler` + `@enforce` decorators, schema binding. No business logic.
- **Decorators**: `@enforce(permission)` handles IAM binding, RBAC, ABAC, rate limiting, and concurrency in one stored procedure call. `@http_error_handler` maps exceptions to HTTP status codes.
- **Schemas**: Pydantic request/response models only. Validation through field constraints, not imperative code.
- **Service**: business logic and validation, orchestrate repositories. No direct DB access, no HTTP concerns.
- **Repository**: UDF/stored procedure calls only. No business logic, no validation. Always call `api.*` UDFs -- never write inline SQL.
- **Utilities** (`utilities/`): pure helpers and domain utilities injected into services. No DB access.
- **Constants** (`constants/`): prompt strings, field codes, config literals. No logic.

### Decorator Order on Every Router Endpoint

```python
@http_error_handler          # outermost - maps exceptions to HTTP
@router.post("/path")
@enforce(permission.name)    # innermost - IAM/rate-limit enforcement
async def endpoint(...):
    pass
```

### UDF Rule

All database queries go through `api.*` UDFs (PostgreSQL functions). Never write inline SQL in repositories. If no UDF exists yet, create one in `sql/chhg/udf/api/` and register it in `database/chhg_udf_load.py`.

### @enforce Authorization Context

After `@enforce` succeeds, the full auth context is available on the request:

```python
request.state.auth.user_id
request.state.auth.role
request.state.auth.plan_id
request.state.auth.organization_id
request.state.auth.project_id
request.state.auth.concurrency_lease_id  # auto-released on completion
```

Access these in the router to pass `user_id`, `project_id`, etc. down to the service. Do not re-derive them from the JWT inside the service layer.

### Container Instantiation Order

Every service under `services/chhg/` is wired by `CHHGServiceContainer`, which holds `CHHGUtilityContainer` and `CHHGRepositoryContainer`. These are the correct types to declare in constructor signatures for all CHHG services.

Never instantiate `CHHGUtilityContainer` or `CHHGRepositoryContainer` directly inside a service body. They must always arrive via constructor injection from `CHHGServiceContainer`.

The correct flow is:

```text
ServiceContainer
  → UtilityContainer.chhg_container  →  CHHGUtilityContainer   ─┐
  → RepositoryContainer.chhg         →  CHHGRepositoryContainer ─┤→ CHHGServiceContainer → each service
```

`UtilityContainer` / `RepositoryContainer` (top-level) must **never** appear in `services/chhg/` constructor signatures — they are parent-level concerns and the CHHG layer never receives them directly.

### Container Lazy-Loading Pattern

Both `RepositoryContainer` and `ServiceContainer` use lazy-loaded properties. Always follow this pattern when adding a new entry:

```python
# In container.py
@property
def my_service(self) -> IMyService:
    if self._my_service is None:
        self._my_service = MyService(repo=self._repo_container)
    return self._my_service
```

Never instantiate services or repositories directly in a router or another service.

---

## SWE Principles

1. **Unidirectional Data Flow** -- data flows in one direction through the layers.
2. **Component Modularity** -- each component has a single, well-defined responsibility.
3. **Zero/Minimal Local State** -- avoid mutable local state; prefer pure functions and explicit data passing.
4. **SOLID Principles** -- Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.
5. **Inversion of Control (IoC)** -- all functions with side effects must use Dependency Injection, Dependency Inversion, Constructor Injection, Composition Root, or Abstract Factory patterns.
6. **Order Functions** -- least dependent (pure/utility) functions at the **top** of each file; most dependent (orchestrating/higher-order) functions at the **bottom**.
7. **Hierarchy from Least to Most Dependent** -- file and module structure reflects the dependency hierarchy.

---

## Dependency Injection Rules

- All components are injected via **interfaces**, not concrete implementations.
- Use **constructor injection** as the default pattern.
- Define a **Composition Root** where all dependencies are wired together.
- Use **Abstract Factory** for dynamic dependency creation.
- Any function with side effects (I/O, DB, network, logging) must receive its dependencies via injection -- never instantiate them internally.

---

## Code Structure Rules

### Function Size

- Functions must be **20-30 lines maximum**.
- If a function exceeds this, decompose it into smaller, single-purpose helpers.

### File Layout

- Least dependent, simplest utility functions go at the **top** of the file.
- Most dependent, orchestrating functions go at the **bottom**.
- Internal helper functions must be **used**. If an internal function is unused, either use it or remove it -- do not leave dead code.

### Type Safety

- Use **Pydantic models** for all data validation and serialization.
- All function signatures must have explicit type annotations.

---

## Error Handling

- **Every function** must have a `try/except` block.
- For missing required inputs: raise `ValueError` **before** checking types. Only check types after confirming all required arguments are present.
- Caught exceptions must be **printed** (or logged).
- Raised exceptions must be **descriptive**: state what crashed, how to fix it, and which invariant was violated.

```python
# Example error message format:
raise ValueError(
    f"[ClassName.method_name] Failed to <do X>. "
    f"Invariant violated: <what was expected>. "
    f"Received: <what was received>. "
    f"Fix: <how to resolve>."
)
```

---

## Logging & Observability

- If `tenant_id`, `workflow_id`, or `session_id` are present in scope, they **must** be included in every log statement to enable execution tracing.
- Logs must be meaningful -- include context, not just a message.

```python
logger.info(
    "Processing started",
    extra={"tenant_id": tenant_id, "workflow_id": workflow_id, "session_id": session_id}
)
```

---

## Docstrings

Every function and class must have a docstring using this format:

```python
def my_function(...):
    """
    #Feature: <high-level feature name>
    #Scenario: <specific scenario this handles>
    #Complexity: <type of complexity - level, e.g., Time O(n) - linear>

    <Brief description of what the function does.>

    Args:
        param1: Description.
        param2: Description.

    Returns:
        Description of return value.

    Raises:
        ValueError: When <condition>.
        RuntimeError: When <condition>.
    """
```

---

## Input Validation Order

1. Check for `None` / missing required arguments -> raise `ValueError` immediately.
2. Check types of confirmed-present arguments.
3. Check value constraints / domain rules.

```python
if param is None:
    raise ValueError("[fn] 'param' is required and must not be None.")
if not isinstance(param, ExpectedType):
    raise TypeError("[fn] 'param' must be ExpectedType, got {type(param)}.")
```

---

## Code Evaluation Checklist

Before finalizing any code, verify:

| Metric | Requirement |
| --- | --- |
| Interface Integrity | Public interfaces (inputs/outputs) unchanged unless intentional |
| SWE Principles Adherence | Class and function design follows all 7 principles |
| Docstrings | Every function/class has `#Feature`, `#Scenario`, `#Complexity` annotations |
| Input Validation & Error Handling | None-check before type-check; descriptive exceptions |
| Edge & Boundary Coverage | All specified edge cases handled |
| Architecture Alignment | Code structure matches the 4-layer architecture |
| Performance & Optimization | Algorithmic efficiency considered; LRU cache used where applicable |
| Readability & Style | Clear naming, consistent style |
| Testability & Mockability | Dependencies injected; clear mock points exist |
| External Dependency Imports | Latest stable versions; all imports used |
| Internal Dependency Imports | Relative paths from root; all imports used |
| Dependency Management | No unused imports; function call signatures match definitions |
| Logging & Observability | tenant_id/workflow_id/session_id included in logs where available |
| Security & Sanitization | Inputs sanitized; no injection vulnerabilities |
| Function Size | No function exceeds 30 lines |
| Dead Code | No unused internal functions |

---

## New API Creation Checklist

When adding a new API endpoint, complete these steps in order:

1. **UDF** - create `sql/chhg/udf/api/<domain>/<name>.sql` using CTEs, not JOINs; register in `chhg_udf_load.py`
2. **Repository interface** - `interfaces/repositories/<domain>/<name>.py` (ABC)
3. **Repository implementation** - `repositories/<domain>/<name>.py` calls the UDF via `text()`
4. **Update `repositories/<domain>/__init__.py`** - export the new class
5. **Add to RepositoryContainer** - lazy property in `repositories/container.py`
6. **Service interface** - `interfaces/services/<domain>/<name>.py` (ABC)
7. **Service implementation** - `services/<domain>/<name>.py` -- business logic only
8. **Update `services/<domain>/__init__.py`** - export the new class
9. **Add to ServiceContainer** - lazy property in `services/container.py`
10. **Pydantic schemas** - `schemas/<domain>/<name>_schemas.py`
11. **Permission constant** - add `resource.action` string to `utilities/auth_reforce/permissions.py`
12. **Router** - `routers/<domain>/<name>.py` with `@http_error_handler` + `@enforce`
13. **Update `routers/<domain>/__init__.py`** - export the router
14. **Register router** - include in `main/app.py`

### `__init__.py` Rule

Every package `__init__.py` must export all public classes and functions from that package using explicit imports and `__all__`. An empty `__init__.py` is a bug.

### Constants Placement

- Prompt strings (system prompts, user templates) -> `constants/chhg/<domain>_prompts.py`
- Field codes (EAV integer constants) -> `constants/chhg/prompt_field_codes.py`
- Type paths, relation names, config literals -> `constants/chhg/<domain>.py`
- Never define multi-line prompt strings inside a service or utility module. Import them from `constants/`.

### Common Router Patterns

```python
# Filtering + pagination
async def list_items(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    services: ServiceContainer = Depends(get_service_container),
):
    ...

# Partial update - only provided fields
async def update_item(request: UpdateRequest, ...):
    await services.domain.update(data=request.model_dump(exclude_unset=True))

# Soft delete - returns message, not the deleted object
async def delete_item(item_id: UUID, ...):
    await services.domain.soft_delete(item_id)
    return {"message": "Deleted successfully"}
```

### Permission Naming Convention

Permissions follow the `resource.action` format. Actions are: `create`, `read`, `update`, `delete`, `manage`, `execute`. Define them as a class in `utilities/auth_reforce/permissions.py` and expose a module-level instance:

```python
class FeaturePermissions:
    create = "feature.create"
    read   = "feature.read"

feature = FeaturePermissions()
```

Use `feature.read` (not the raw string) at the call site so renames propagate automatically.

---

## Error Taxonomy

Use `errors/errors.py` exclusively. Never raise raw `HTTPException`.

| Error class | When to use |
| --- | --- |
| `Errors.HTTP.BadRequest` | Missing/invalid request params |
| `Errors.HTTP.Unauthorized` | No/invalid JWT |
| `Errors.HTTP.Forbidden` | Valid JWT, insufficient permission |
| `Errors.HTTP.NotFound` | Resource does not exist |
| `Errors.HTTP.InternalServerError` | Unexpected DB or system failure |
| `Errors.Database.IntegrityError` | Unique constraint violation |
| `Errors.Misc.*` | Domain-specific business rule violations |

---

## Summary of Patterns to Always Use

- **Pydantic** for models and validation
- **Abstract interfaces** (ABC or Protocol) for all injectable dependencies
- **Constructor injection** for all components
- **LRU cache** (configurable TTL) for expensive repeated computations
- **Proper exception hierarchy** -- define domain-specific exceptions where appropriate
- **BDD-style docstrings** (`#Feature`, `#Scenario`) on every function and class
- **`api.*` UDFs** for all database access -- no inline SQL in repositories
- **CTEs over JOINs** when writing SQL UDFs
- **`@http_error_handler` before `@enforce`** on every router function
- **Explicit `__init__.py` exports** -- never leave a package `__init__.py` empty
