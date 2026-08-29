// tfjs-node 4.22.0 calls util.isNullOrUndefined, a Node builtin deprecated for years
// and finally removed in Node 23. This project runs on Node 26, so the call throws
// "util_1.isNullOrUndefined is not a function" on the first cast op.
//
// The original implementation was exactly this one-liner, so restoring it is safe.
// MUST be imported before @tensorflow/tfjs-node. ES module imports are evaluated in
// declaration order, so importing this first in a module is sufficient.

import util from 'node:util';

type LegacyUtil = typeof util & {
  isNullOrUndefined?: (arg: unknown) => boolean;
};

const legacy = util as LegacyUtil;

if (typeof legacy.isNullOrUndefined !== 'function') {
  legacy.isNullOrUndefined = (arg: unknown): boolean => arg === null || arg === undefined;
}
