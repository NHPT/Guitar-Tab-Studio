declare module 'plist' {
  export function build(value: unknown): string
  export function parse(xml: string): unknown

  const plist: {
    build: typeof build
    parse: typeof parse
  }

  export default plist
}
