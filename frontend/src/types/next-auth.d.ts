import type { DefaultSession } from "next-auth";
import type { JWT as DefaultJWT } from "next-auth/jwt";

declare module "next-auth" {
  interface Session {
    user?: DefaultSession["user"];
    /**
     * Google ID token forwarded to the FastAPI backend.
     */
    idToken?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT extends DefaultJWT {
    /**
     * Google ID token captured during the OAuth flow.
     */
    idToken?: string;
  }
}

