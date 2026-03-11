import NextAuth, { type DefaultSession } from "next-auth";
import Google from "next-auth/providers/google";

// 1. Updated Module Augmentation for Auth.js v5
declare module "next-auth" {
  interface Session {
    idToken?: string;
    user?: {
      email: string; // Ensure email is required in the session user
    } & DefaultSession["user"];
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  pages: {
    error: "/auth/error",
  },
  providers: [
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
      authorization: {
        params: {
          scope: "openid email profile",
          prompt: "select_account",
          access_type: "offline",
          response_type: "code"
        },
      },
    }),
  ],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    async signIn({ profile }) {
      const email = profile?.email ?? "";
      return email.endsWith("@berkeley.edu");
    },

    async jwt({ token, account, profile }) {
      // 2. Persist email from profile into the JWT token for proxy.ts access
      if (profile?.email) {
        token.email = profile.email;
      }
      if (account?.id_token) {
        token.idToken = account.id_token;
      }
      return token;
    },

    async session({ session, token }) {
      // 3. Sync the token data to the session object
      if (token.idToken) {
        session.idToken = token.idToken as string;
      }
      if (token.email && session.user) {
        session.user.email = token.email as string;
      }
      return session;
    },
  },
});