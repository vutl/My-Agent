// The cubism4 entry point shares the root package's types; "Node" module
// resolution does not read the package's subpath exports.
declare module "pixi-live2d-display/cubism4" {
  export * from "pixi-live2d-display";
}
