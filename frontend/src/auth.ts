import NextAuth, { type DefaultSession } from "next-auth";
import Google from "next-auth/providers/google";

// This tells TypeScript that 'session' includes our idToken
declare module "next-auth" {
  interface Session {
    idToken?: string;
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
          prompt: "select_account", // This forces Google to show the account picker
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
    /**
     * Berkeley-only gate: only allow @berkeley.edu emails to sign in.
     */
    async signIn({ profile }) {
      const email = (profile as { email?: string } | null)?.email ?? "";
      return email.endsWith("@berkeley.edu");
    },

    /**
     * Token callback: capture Google's ID token on initial sign-in
     * so we can forward it to the FastAPI backend.
     */
    async jwt({ token, account, profile }) {
      const email = (profile as { email?: string } | null)?.email ?? undefined;
      if (email) token.email = email;
      if (account) token.idToken = account.id_token;
      return token;
    },

    /**
     * Session callback: expose the ID token to the client session
     * so the chat UI can send it as a Bearer token to FastAPI.
     */
    async session({ session, token }) {
      // Directly assign the property
      if (token.idToken) {
        session.idToken = token.idToken as string;
      }
      return session;
    },
  },
});

