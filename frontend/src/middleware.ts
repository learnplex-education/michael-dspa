import { withAuth } from "next-auth/middleware";

export default withAuth({
  callbacks: {
    authorized: ({ token }) => {
      const email = (token?.email as string | undefined) ?? "";
      return email.endsWith("@berkeley.edu");
    },
  },
  pages: {
    error: "/auth/error",
    signIn: "/api/auth/signin",
  },
});

export const config = {
  matcher: [
    // Protect the chat experience. If you later move chat to /chat, it will already be guarded.
    "/",
    "/chat/:path*",
  ],
};

