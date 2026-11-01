#include "speculative.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    common_speculative_backoff policy;
    for (int i = 0; i < 7; ++i) {
        assert(policy.should_draft());
        policy.accept(0);
    }
    assert(policy.should_draft());
    policy.accept(1);
    for (int i = 0; i < 8; ++i) {
        assert(policy.should_draft());
        policy.accept(0);
    }
    for (int i = 0; i < 32; ++i) {
        assert(!policy.should_draft());
        policy.accept(0);
    }
    assert(policy.should_draft());
    policy.accept(0);
    for (int i = 0; i < 32; ++i) {
        assert(!policy.should_draft());
    }
    assert(policy.should_draft());
    policy.accept(2);
    assert(policy.should_draft());
    policy.accept(0);
    assert(policy.should_draft());

    common_speculative_backoff other;
    for (int i = 0; i < 8; ++i) {
        assert(other.should_draft());
        other.accept(0);
    }
    assert(!other.should_draft());
    assert(policy.should_draft());
    other = {};
    assert(other.should_draft());
}
